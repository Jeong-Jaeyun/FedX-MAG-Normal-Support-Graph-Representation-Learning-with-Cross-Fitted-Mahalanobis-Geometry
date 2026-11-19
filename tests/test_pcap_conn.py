import csv
import struct

from fedxmag.pcap_conn import FIELDS, track_pcapng, write_conn_csv
from fedxmag.ton_iot import FLAG_STATES, read_ton_records

FIN, SYN, RST, PSH, ACK = 0x01, 0x02, 0x04, 0x08, 0x10
CLIENT, SERVER = "192.168.10.5", "192.168.10.50"
T0 = 1_499_342_400  # 2017-07-06 12:00:00 UTC


def _ip(address):
    return bytes(int(part) for part in address.split("."))


def _ipv4(source, destination, transport, body):
    header = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(body), 0, 0, 64, transport, 0,
                         _ip(source), _ip(destination))
    return b"\x00" * 12 + b"\x08\x00" + header + body


def tcp(source, sport, destination, dport, flags, seq=0, payload=b""):
    body = struct.pack("!HHIIBBHHH", sport, dport, seq, 0, 5 << 4, flags, 1024, 0, 0) + payload
    return _ipv4(source, destination, 6, body)


def udp(source, sport, destination, dport, payload=b""):
    body = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload
    return _ipv4(source, destination, 17, body)


def icmp(source, destination, kind, payload=b""):
    return _ipv4(source, destination, 1, struct.pack("!BBHHH", kind, 0, 0, 1, 1) + payload)


def _block(kind, body):
    return struct.pack("<II", kind, 12 + len(body)) + body + struct.pack("<I", 12 + len(body))


def write_pcapng(path, packets):
    blob = _block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
    blob += _block(1, struct.pack("<HHI", 1, 0, 65535))
    for seconds, frame in packets:
        micros = int(round(seconds * 1_000_000))
        padded = frame + b"\x00" * (-len(frame) % 4)
        blob += _block(6, struct.pack("<IIIII", 0, micros >> 32, micros & 0xFFFFFFFF,
                                      len(frame), len(frame)) + padded)
    path.write_bytes(blob)


def connections(tmp_path, packets):
    write_pcapng(tmp_path / "capture.pcapng", packets)
    records = track_pcapng(tmp_path / "capture.pcapng").close()
    return [dict(zip(FIELDS, record, strict=True)) for record in records]


def test_tcp_handshake_data_and_close(tmp_path):
    request, response = b"GET / HTTP/1.1\r\n\r\n", b"HTTP/1.1 200 OK\r\n\r\nhello"
    (record,) = connections(tmp_path, [
        (T0, tcp(CLIENT, 50000, SERVER, 80, SYN, seq=100)),
        (T0 + 0.01, tcp(SERVER, 80, CLIENT, 50000, SYN | ACK, seq=900)),
        (T0 + 0.02, tcp(CLIENT, 50000, SERVER, 80, ACK, seq=101)),
        (T0 + 0.03, tcp(CLIENT, 50000, SERVER, 80, PSH | ACK, seq=101, payload=request)),
        (T0 + 0.04, tcp(CLIENT, 50000, SERVER, 80, PSH | ACK, seq=101, payload=request)),
        (T0 + 0.05, tcp(SERVER, 80, CLIENT, 50000, PSH | ACK, seq=901, payload=response)),
        (T0 + 0.06, tcp(CLIENT, 50000, SERVER, 80, FIN | ACK, seq=101 + len(request))),
        (T0 + 0.07, tcp(SERVER, 80, CLIENT, 50000, FIN | ACK, seq=901 + len(response))),
    ])
    assert (record["src_ip"], record["src_port"], record["dst_ip"], record["dst_port"]) == (
        CLIENT, 50000, SERVER, 80)
    assert record["conn_state"] == "SF" and record["service"] == "http"
    # The retransmitted request counts once in bytes but twice in packets.
    assert (record["src_bytes"], record["dst_bytes"]) == (len(request), len(response))
    assert (record["src_pkts"], record["dst_pkts"]) == (5, 3)
    assert record["duration"] == 70_000_000


def test_tcp_attempt_states(tmp_path):
    records = connections(tmp_path, [
        (T0, tcp(CLIENT, 40001, SERVER, 22, SYN)),                       # S0
        (T0, tcp(CLIENT, 40002, SERVER, 23, SYN)),                       # REJ
        (T0 + 0.01, tcp(SERVER, 23, CLIENT, 40002, RST | ACK)),
        (T0, tcp(CLIENT, 40003, SERVER, 443, SYN)),                      # RSTO
        (T0 + 0.01, tcp(SERVER, 443, CLIENT, 40003, SYN | ACK)),
        (T0 + 0.02, tcp(CLIENT, 40003, SERVER, 443, RST)),
        (T0, tcp(CLIENT, 40004, SERVER, 8080, SYN)),                     # RSTOS0
        (T0 + 0.02, tcp(CLIENT, 40004, SERVER, 8080, RST)),
        (T0, tcp(SERVER, 80, CLIENT, 40005, SYN | ACK)),                 # flipped
    ])
    state = {r["src_port"]: r["conn_state"] for r in records}
    assert state == {40001: "S0", 40002: "REJ", 40003: "RSTO", 40004: "RSTOS0", 40005: "OTH"}
    flipped = next(r for r in records if r["src_port"] == 40005)
    assert (flipped["src_ip"], flipped["dst_ip"], flipped["dst_port"]) == (CLIENT, SERVER, 80)
    assert all(r["service"] == "-" for r in records)


def test_port_reuse_and_timeouts(tmp_path):
    records = connections(tmp_path, [
        (T0, tcp(CLIENT, 41000, SERVER, 80, SYN)),
        (T0 + 0.1, tcp(SERVER, 80, CLIENT, 41000, RST | ACK)),
        (T0 + 1, tcp(CLIENT, 41000, SERVER, 80, SYN)),                   # reuse after reset
        (T0, udp(CLIENT, 45000, SERVER, 53, b"\x12\x34query")),
        (T0 + 0.01, udp(SERVER, 53, CLIENT, 45000, b"\x12\x34answer")),
        (T0 + 61.5, udp(CLIENT, 45000, SERVER, 53, b"\x12\x35query")),   # after UDP timeout
        (T0 + 2, udp(SERVER, 53, CLIENT, 45001, b"\x00late")),            # reply first: flip
    ])
    tcp_states = [r["conn_state"] for r in records if r["proto"] == "tcp"]
    assert tcp_states == ["REJ", "S0"]
    dns = [r for r in records if r["proto"] == "udp"]
    assert [(r["src_port"], r["conn_state"], r["service"]) for r in dns] == [
        (45000, "SF", "dns"), (45001, "SHR", "dns"), (45000, "S0", "dns")]
    assert dns[0]["src_bytes"] == 7 and dns[0]["dst_bytes"] == 8


def test_icmp_echo_pairs_with_reply(tmp_path):
    (record,) = connections(tmp_path, [
        (T0, icmp(CLIENT, SERVER, 8, b"ping")),
        (T0 + 0.001, icmp(SERVER, CLIENT, 0, b"ping")),
    ])
    assert (record["proto"], record["src_port"], record["dst_port"]) == ("icmp", 8, 0)
    assert (record["conn_state"], record["src_pkts"], record["dst_pkts"]) == ("OTH", 1, 1)
    assert (record["src_bytes"], record["dst_bytes"]) == (4, 4)


def test_conn_csv_reads_through_frozen_ton_adapter(tmp_path):
    write_pcapng(tmp_path / "capture.pcapng", [
        (T0, tcp(CLIENT, 40001, SERVER, 22, SYN)),
        (T0 + 0.5, tcp(CLIENT, 40003, SERVER, 443, SYN)),
        (T0 + 0.51, tcp(SERVER, 443, CLIENT, 40003, SYN | ACK)),
        (T0 + 0.52, tcp(CLIENT, 40003, SERVER, 443, ACK)),
        (T0 + 0.53, tcp(CLIENT, 40003, SERVER, 443, PSH | ACK, payload=b"\x16\x03\x01\x00")),
        (T0 + 1, udp(CLIENT, 45000, SERVER, 53, b"query")),
    ])
    write_conn_csv(track_pcapng(tmp_path / "capture.pcapng").close(), tmp_path / "conn.csv")
    with (tmp_path / "conn.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == list(FIELDS)
    with (tmp_path / "labelled.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*rows[0], "label", "type"])
        writer.writerows([*row, "0", "normal"] for row in rows[1:])
    records = read_ton_records([tmp_path / "labelled.csv"])
    assert records.report["kept"] == 3
    assert records.timestamps.tolist() == [s * 1_000_000_000 for s in (T0, T0, T0 + 1)]
    states = [records.state_names[c] for c in records.states]
    services = [records.service_names[c] for c in records.services]
    assert states == ["S0", "S1", "S0"] and services == [None, "ssl", "dns"]
    flows = records.flow_records(range(3))
    assert flows.services == ["other", "ssl", "dns"]
    assert "S1" in FLAG_STATES["Is_SYN_ACK"]
