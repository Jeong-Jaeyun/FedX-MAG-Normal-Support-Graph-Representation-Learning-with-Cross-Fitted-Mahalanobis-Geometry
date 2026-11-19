"""Zeek-style connection records from pcapng captures, without labels.

Packets are grouped into bidirectional transport connections and summarized with the
Zeek conn.log fields that :func:`fedxmag.ton_iot.read_ton_records` reads, so a capture
can be featurized exactly like TON_IoT once label columns are added. This is our own
implementation of Zeek's documented semantics, not Zeek:

- key: protocol and the unordered endpoint pair (address, port); ICMP uses the
  unordered address pair and the request type, so echo, timestamp, information, and
  address-mask requests share a connection with their replies;
- originator: the first packet's sender, except that a first TCP SYN-ACK, a first ICMP
  reply, or a first non-SYN packet from a likely server port to a port that is not one
  makes the receiver the originator;
- end: 300 s (TCP) or 60 s (UDP, ICMP) without packets; a TCP connection closed by FIN
  in both directions or reset by either side also ends at the next SYN without ACK or
  after 5 s without packets;
- bytes: TCP payload by sequence-number span (retransmissions count once), UDP and
  ICMP payload lengths summed; packets: every IP packet of the connection;
- conn_state: Zeek's rules over per-endpoint TCP states (inactive, SYN sent, SYN-ACK
  sent, partial, established, closed, reset); an originator becomes established when
  it acknowledges an established responder. UDP is SF, S0, SHR, or OTH; ICMP is OTH;
- service: ``-`` without payload; else an HTTP, TLS, or SSH signature at the start of
  either side's first payload; else the analyzer registered for the responder port;
  else ``-``.

Non-IP frames, non-first IP fragments, truncated headers, and transports other than
TCP, UDP, and ICMP are counted and skipped. ``ts`` is the first packet time floored to
whole seconds, the TON_IoT resolution.
"""

from __future__ import annotations

import csv
import ipaddress
import mmap
import struct
from collections import Counter
from pathlib import Path
from typing import Any

FIELDS = (
    "ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service", "duration",
    "src_bytes", "dst_bytes", "conn_state", "src_pkts", "dst_pkts",
)
SECOND_NS = 1_000_000_000
TIMEOUT_NS = {"tcp": 300 * SECOND_NS, "udp": 60 * SECOND_NS, "icmp": 60 * SECOND_NS}
TCP_CLOSE_DELAY_NS = 5 * SECOND_NS
SWEEP_NS = 60 * SECOND_NS
SEQ_MASK = 0xFFFFFFFF
FIN, SYN, RST, ACK = 0x01, 0x02, 0x04, 0x10
INACTIVE, SYN_SENT, SYN_ACK_SENT, PARTIAL, ESTABLISHED, CLOSED, RESET = range(7)

LIKELY_SERVER_PORTS = {
    "tcp": frozenset({
        21, 22, 23, 25, 53, 80, 81, 88, 110, 111, 135, 139, 143, 389, 443, 445, 465, 502,
        563, 587, 631, 636, 993, 995, 1080, 1433, 1521, 1723, 1883, 3128, 3306, 3389, 5060,
        5222, 5432, 5900, 6379, 6667, 8000, 8080, 8443, 8888, 9200, 20000, 27017,
    }),
    "udp": frozenset({
        53, 67, 69, 88, 123, 137, 138, 161, 162, 500, 514, 1812, 1813, 1900, 3478, 4500,
        5060, 5353, 5355, 47808,
    }),
}
PORT_SERVICES = {
    **{("tcp", port): name for port, name in (
        (21, "ftp"), (22, "ssh"), (25, "smtp"), (53, "dns"), (88, "krb_tcp"), (110, "pop3"),
        (135, "dce_rpc"), (139, "smb"), (143, "imap"), (389, "ldap"), (445, "smb"),
        (502, "modbus"), (587, "smtp"), (1883, "mqtt"), (3306, "mysql"), (3389, "rdp"),
        (6667, "irc"), (20000, "dnp3"),
    )},
    **{("udp", port): name for port, name in (
        (53, "dns"), (67, "dhcp"), (68, "dhcp"), (88, "krb"), (123, "ntp"), (137, "dns"),
        (161, "snmp"), (162, "snmp"), (1812, "radius"), (5060, "sip"), (5353, "dns"),
        (5355, "dns"),
    )},
}
HTTP_PREFIXES = (
    b"GET ", b"POST ", b"HEAD ", b"PUT ", b"DELETE ", b"OPTIONS ", b"CONNECT ", b"PATCH ",
    b"TRACE ",
)
# ICMP request type of each request or reply type (ICMPv4 and ICMPv6).
ICMP_REQUEST = {0: 8, 8: 8, 13: 13, 14: 13, 15: 15, 16: 15, 17: 17, 18: 17, 128: 128, 129: 128}
ICMP_REPLIES = frozenset({0, 14, 16, 18, 129})
IPV6_EXTENSIONS = frozenset({0, 43, 44, 60})


class _Conn:
    __slots__ = ("proto", "start", "last", "orig", "resp", "pkts", "sizes", "base", "state",
                 "clean_syn", "payload")

    def __init__(self, proto: str, start: int, orig: tuple, resp: tuple) -> None:
        self.proto, self.start, self.last, self.orig, self.resp = proto, start, start, orig, resp
        self.pkts = [0, 0]
        self.sizes = [0, 0]  # TCP: highest sequence offset; UDP and ICMP: summed payload
        self.base: list[int | None] = [None, None]
        self.state = [INACTIVE, INACTIVE]
        self.clean_syn: bool | None = None
        self.payload = [b"", b""]

    def terminated(self) -> bool:
        return RESET in self.state or self.state == [CLOSED, CLOSED]


def conn_state(conn: _Conn) -> str:
    if conn.proto == "udp":
        orig, resp = conn.pkts[0] > 0, conn.pkts[1] > 0
        return ("SF" if resp else "S0") if orig else ("SHR" if resp else "OTH")
    if conn.proto != "tcp":
        return "OTH"
    os, rs = conn.state
    o_inactive, r_inactive = os in (INACTIVE, PARTIAL), rs in (INACTIVE, PARTIAL)
    if rs == RESET:
        if os in (SYN_SENT, SYN_ACK_SENT) or (os == RESET and conn.sizes == [0, 0]):
            return "REJ"
        return "RSTRH" if o_inactive else "RSTR"
    if os == RESET:
        if r_inactive:
            return "RSTOS0" if conn.clean_syn else "OTH"
        return "RSTO"
    if os == CLOSED and rs == CLOSED:
        return "SF"
    if os == CLOSED:
        return "SH" if r_inactive else "S2"
    if rs == CLOSED:
        return "SHR" if o_inactive else "S3"
    if os == SYN_SENT and rs == INACTIVE:
        return "S0"
    if os == ESTABLISHED and rs == ESTABLISHED:
        return "S1"
    return "OTH"


def service(conn: _Conn) -> str:
    orig, resp = conn.payload
    if not orig and not resp:
        return "-"
    if orig.startswith(HTTP_PREFIXES) or resp.startswith(b"HTTP/"):
        return "http"
    for prefix in (orig, resp):
        if len(prefix) >= 3 and prefix[0] == 0x16 and prefix[1] == 0x03 and prefix[2] <= 4:
            return "ssl"
    if orig.startswith(b"SSH-") or resp.startswith(b"SSH-"):
        return "ssh"
    return PORT_SERVICES.get((conn.proto, conn.resp[1]), "-")


class ConnTracker:
    """Packets in capture order in, finished connection records out."""

    def __init__(self) -> None:
        self.active: dict[tuple, _Conn] = {}
        self.records: list[tuple] = []
        self.counts: Counter[str] = Counter()
        self.next_sweep: int | None = None

    def _finish(self, key: tuple, conn: _Conn) -> None:
        del self.active[key]
        tcp = conn.proto == "tcp"
        self.records.append((
            conn.start, conn.orig[0], conn.orig[1], conn.resp[0], conn.resp[1], conn.proto,
            service(conn), conn.last - conn.start, conn.sizes[0], conn.sizes[1],
            conn_state(conn), conn.pkts[0], conn.pkts[1],
        ))
        self.counts[f"connections_{conn.proto}"] += 1
        if tcp:
            self.counts["connections_tcp_terminated"] += conn.terminated()

    def _expired(self, conn: _Conn, now: int) -> bool:
        idle = now - conn.last
        return idle > TIMEOUT_NS[conn.proto] or (
            conn.proto == "tcp" and conn.terminated() and idle > TCP_CLOSE_DELAY_NS
        )

    def _sweep(self, now: int) -> None:
        if self.next_sweep is None:
            self.next_sweep = now + SWEEP_NS
        if now < self.next_sweep:
            return
        for key, conn in list(self.active.items()):
            if self._expired(conn, now):
                self._finish(key, conn)
        self.next_sweep = now + SWEEP_NS

    def _payload(self, conn: _Conn, side: int, length: int, data: Any, offset: int) -> None:
        if length > 0 and not conn.payload[side]:
            conn.payload[side] = bytes(data[offset : offset + min(length, 8)])

    def tcp(self, now: int, a: tuple, b: tuple, flags: int, seq: int, length: int,
            data: Any, offset: int) -> None:
        self._sweep(now)
        key = ("tcp", a, b) if a <= b else ("tcp", b, a)
        conn = self.active.get(key)
        syn, ack, fin, rst = flags & SYN, flags & ACK, flags & FIN, flags & RST
        if conn is not None and (
            self._expired(conn, now) or (conn.terminated() and syn and not ack)
        ):
            self._finish(key, conn)
            conn = None
        if conn is None:
            server = LIKELY_SERVER_PORTS["tcp"]
            flip = bool(syn and ack) or (
                not syn and a[1] in server and b[1] not in server
            )
            conn = _Conn("tcp", now, b, a) if flip else _Conn("tcp", now, a, b)
            self.active[key] = conn
        side = 0 if a == conn.orig else 1
        conn.last = now
        conn.pkts[side] += 1
        if side == 0 and not rst:
            pure_syn = bool(syn and not ack and not fin)
            conn.clean_syn = pure_syn if conn.clean_syn is None else conn.clean_syn and (
                pure_syn or not (ack or fin)
            )
        if syn and conn.base[side] is None:
            conn.base[side] = (seq + 1) & SEQ_MASK
        if length > 0:
            start = (seq + (1 if syn else 0)) & SEQ_MASK
            if conn.base[side] is None:
                conn.base[side] = start
            relative = (start - conn.base[side]) & SEQ_MASK
            if relative < 0x80000000:
                conn.sizes[side] = max(conn.sizes[side], relative + length)
            self._payload(conn, side, length, data, offset)
        state, peer = conn.state[side], conn.state[1 - side]
        if rst:
            state = RESET
        else:
            if state == INACTIVE:
                if syn:
                    if side == 0:
                        state = SYN_ACK_SENT if ack else SYN_SENT
                    else:
                        state = ESTABLISHED if ack else SYN_SENT
                else:
                    state = PARTIAL
            elif state in (SYN_SENT, SYN_ACK_SENT) and (
                (syn and ack and side == 1) or (ack and not syn and peer in (ESTABLISHED, CLOSED))
            ):
                state = ESTABLISHED
            if fin and state != RESET:
                state = CLOSED
        conn.state[side] = state

    def udp(self, now: int, a: tuple, b: tuple, length: int, data: Any, offset: int) -> None:
        self._sweep(now)
        key = ("udp", a, b) if a <= b else ("udp", b, a)
        conn = self.active.get(key)
        if conn is not None and self._expired(conn, now):
            self._finish(key, conn)
            conn = None
        if conn is None:
            server = LIKELY_SERVER_PORTS["udp"]
            flip = a[1] in server and b[1] not in server
            conn = _Conn("udp", now, b, a) if flip else _Conn("udp", now, a, b)
            self.active[key] = conn
        side = 0 if a == conn.orig else 1
        conn.last = now
        conn.pkts[side] += 1
        conn.sizes[side] += max(length, 0)
        self._payload(conn, side, length, data, offset)

    def icmp(self, now: int, source: str, destination: str, kind: int, code: int,
             length: int) -> None:
        self._sweep(now)
        request = ICMP_REQUEST.get(kind, kind)
        pair = (source, destination) if source <= destination else (destination, source)
        key = ("icmp", *pair, request)
        conn = self.active.get(key)
        if conn is not None and self._expired(conn, now):
            self._finish(key, conn)
            conn = None
        if conn is None:
            if kind in ICMP_REPLIES:
                conn = _Conn("icmp", now, (destination, request), (source, code))
            else:
                conn = _Conn("icmp", now, (source, request), (destination, code))
            self.active[key] = conn
        side = 0 if source == conn.orig[0] else 1
        conn.last = now
        conn.pkts[side] += 1
        conn.sizes[side] += max(length, 0)

    def close(self) -> list[tuple]:
        for key, conn in list(self.active.items()):
            self._finish(key, conn)
        self.records.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[5]))
        return self.records


def _resolutions(data: Any, start: int, length: int) -> int:
    """Nanoseconds per timestamp unit from an interface description block."""

    position, end = start + 16, start + length - 4
    while position + 4 <= end:
        code, size = struct.unpack_from("<HH", data, position)
        if code == 0:
            break
        if code == 9:
            value = data[position + 4]
            unit = 2.0 ** -(value & 0x7F) if value & 0x80 else 10.0 ** -value
            return max(1, round(unit * SECOND_NS))
        position += 4 + (size + 3) // 4 * 4
    return 1_000  # default resolution: microseconds


def track_pcapng(path: str | Path, tracker: ConnTracker | None = None) -> ConnTracker:
    """Feed every Ethernet frame of a little-endian pcapng file to a tracker."""

    tracker = tracker or ConnTracker()
    counts = tracker.counts
    unpack = struct.unpack_from
    names: dict[bytes, str] = {}

    def address(raw: bytes) -> str:
        name = names.get(raw)
        if name is None:
            name = names[raw] = str(ipaddress.ip_address(raw))
        return name

    with Path(path).open("rb") as stream, mmap.mmap(
        stream.fileno(), 0, access=mmap.ACCESS_READ
    ) as data:
        size, position = len(data), 0
        interfaces: list[tuple[int, int]] = []
        while position + 12 <= size:
            kind, length = unpack("<II", data, position)
            if length < 12 or position + length > size:
                counts["truncated_block"] += 1
                break
            if kind == 0x0A0D0D0A:
                if unpack("<I", data, position + 8)[0] != 0x1A2B3C4D:
                    raise ValueError("Only little-endian pcapng sections are supported")
                interfaces = []
            elif kind == 1:
                interfaces.append((unpack("<H", data, position + 8)[0],
                                   _resolutions(data, position, length)))
            elif kind == 6:
                interface, high, low, captured = unpack("<IIII", data, position + 8)
                link, scale = interfaces[interface]
                now = ((high << 32) | low) * scale
                counts["packets"] += 1
                if link != 1:
                    counts["skipped_link_type"] += 1
                else:
                    _frame(tracker, counts, data, position + 28, captured, now, address, unpack)
            else:
                counts[f"block_{kind}"] += 1
            position += length
    return tracker


def _frame(tracker: ConnTracker, counts: Counter, data: Any, start: int, captured: int,
           now: int, address: Any, unpack: Any) -> None:
    end = start + captured
    if captured < 14:
        counts["skipped_truncated"] += 1
        return
    ether = unpack("!H", data, start + 12)[0]
    l3 = start + 14
    while ether in (0x8100, 0x88A8) and l3 + 4 <= end:
        ether = unpack("!H", data, l3 + 2)[0]
        l3 += 4
    if ether == 0x0800:
        if l3 + 20 > end:
            counts["skipped_truncated"] += 1
            return
        header = (data[l3] & 0x0F) * 4
        total, fragment = unpack("!H", data, l3 + 2)[0], unpack("!H", data, l3 + 6)[0]
        if fragment & 0x1FFF:
            counts["skipped_fragment"] += 1
            return
        transport = data[l3 + 9]
        source, destination = address(data[l3 + 12 : l3 + 16]), address(data[l3 + 16 : l3 + 20])
        l4, remaining = l3 + header, total - header
    elif ether == 0x86DD:
        if l3 + 40 > end:
            counts["skipped_truncated"] += 1
            return
        remaining, transport = unpack("!H", data, l3 + 4)[0], data[l3 + 6]
        source, destination = address(data[l3 + 8 : l3 + 24]), address(data[l3 + 24 : l3 + 40])
        l4 = l3 + 40
        while transport in IPV6_EXTENSIONS and l4 + 8 <= end:
            if transport == 44:
                if unpack("!H", data, l4 + 2)[0] >> 3:
                    counts["skipped_fragment"] += 1
                    return
                step = 8
            else:
                step = (data[l4 + 1] + 1) * 8
            transport = data[l4]
            l4 += step
            remaining -= step
    else:
        counts["skipped_non_ip"] += 1
        return
    if transport == 6:
        if l4 + 20 > end:
            counts["skipped_truncated"] += 1
            return
        sport, dport, seq = unpack("!HHI", data, l4)
        offset = (data[l4 + 12] >> 4) * 4
        tracker.tcp(now, (source, sport), (destination, dport), data[l4 + 13], seq,
                    remaining - offset, data, l4 + offset)
    elif transport == 17:
        if l4 + 8 > end:
            counts["skipped_truncated"] += 1
            return
        sport, dport = unpack("!HH", data, l4)
        tracker.udp(now, (source, sport), (destination, dport), remaining - 8, data, l4 + 8)
    elif transport in (1, 58):
        if l4 + 8 > end:
            counts["skipped_truncated"] += 1
            return
        tracker.icmp(now, source, destination, data[l4], data[l4 + 1], remaining - 8)
    else:
        counts["skipped_transport"] += 1


def write_conn_csv(records: list[tuple], destination: str | Path) -> None:
    """Write records in TON_IoT column conventions (ts in whole seconds)."""

    with Path(destination).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for (start, src, sport, dst, dport, proto, svc, duration, sbytes, dbytes, state,
             spkts, dpkts) in records:
            writer.writerow((
                start // SECOND_NS, src, sport, dst, dport, proto, svc,
                f"{duration / SECOND_NS:.6f}", sbytes, dbytes, state, spkts, dpkts,
            ))
