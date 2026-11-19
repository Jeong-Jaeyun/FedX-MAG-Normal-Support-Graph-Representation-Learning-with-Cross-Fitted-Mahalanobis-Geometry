"""Label-free HTTP observations joined to the existing connection samples.

The archived tracker keeps only eight payload bytes to recognize a service.
This observer preserves its connection boundaries and counters, but separately
reassembles up to 16 KiB of each TCP direction to describe HTTP request targets
and response status. It reads no label, attack schedule, or attacker address.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import mmap
import re
import struct
from pathlib import Path
from urllib.parse import unquote_to_bytes

import numpy as np

from .experiments import sha256, write_json

HTTP_FEATURES = (
    "http_observed",
    "log_request_count",
    "log_uri_length_max",
    "uri_percent_fraction_max",
    "uri_punctuation_fraction_max",
    "decoded_uri_punctuation_fraction_max",
    "log_query_parameter_count_max",
    "post_fraction",
    "get_fraction",
    "log_response_count",
    "response_4xx_fraction",
    "response_5xx_fraction",
    "response_401_fraction",
    "response_403_fraction",
    "log_request_header_length_max",
    "log_cookie_length_max",
    "log_declared_content_length_max",
    "stream_truncated",
)
REQUEST = re.compile(
    rb"(?:^|\r\n)(GET|POST|HEAD|PUT|DELETE|OPTIONS|CONNECT|PATCH|TRACE) "
    rb"([^\r\n ]+) HTTP/1\.[01]\r\n"
)
RESPONSE = re.compile(rb"(?:^|\r\n)HTTP/1\.[01] ([0-9]{3})(?: |\r\n)")
PUNCTUATION = frozenset(b"<>;'\"(){}")


def describe_http(origin: bytes, responder: bytes, truncated: bool = False) -> np.ndarray:
    requests = list(REQUEST.finditer(origin))
    responses = [int(match.group(1)) for match in RESPONSE.finditer(responder)]
    result = np.zeros(len(HTTP_FEATURES), dtype=np.float32)
    if not requests and not responses:
        return result
    result[0] = 1
    result[1] = np.log1p(len(requests))
    result[9] = np.log1p(len(responses))
    result[17] = float(truncated)
    if requests:
        uris = [match.group(2) for match in requests]
        result[2] = np.log1p(max(map(len, uris)))
        result[3] = max(uri.count(b"%") / max(len(uri), 1) for uri in uris)
        result[4] = max(sum(byte in PUNCTUATION for byte in uri) / max(len(uri), 1) for uri in uris)
        decoded = [unquote_to_bytes(uri) for uri in uris]
        result[5] = max(
            sum(byte in PUNCTUATION for byte in uri) / max(len(uri), 1) for uri in decoded
        )
        result[6] = np.log1p(max(uri.count(b"&") + (b"?" in uri) for uri in uris))
        result[7] = sum(match.group(1) == b"POST" for match in requests) / len(requests)
        result[8] = sum(match.group(1) == b"GET" for match in requests) / len(requests)
        header_lengths, cookie_lengths, content_lengths = [], [], []
        for match in requests:
            end = origin.find(b"\r\n\r\n", match.start())
            if end < 0:
                continue
            header = origin[match.start() : end + 4]
            header_lengths.append(len(header))
            cookie = re.search(rb"(?im)^cookie: *([^\r\n]*)", header)
            if cookie:
                cookie_lengths.append(len(cookie.group(1)))
            content = re.search(rb"(?im)^content-length: *([0-9]+)", header)
            if content:
                content_lengths.append(min(int(content.group(1)), 10**12))
        result[14] = np.log1p(max(header_lengths, default=0))
        result[15] = np.log1p(max(cookie_lengths, default=0))
        result[16] = np.log1p(max(content_lengths, default=0))
    if responses:
        result[10] = sum(400 <= status < 500 for status in responses) / len(responses)
        result[11] = sum(500 <= status < 600 for status in responses) / len(responses)
        result[12] = responses.count(401) / len(responses)
        result[13] = responses.count(403) / len(responses)
    return result


def extract(path: Path, *, progress=None) -> tuple[list[tuple], dict[tuple, np.ndarray], dict]:
    # Reuse the local conversion mechanics read-only. The project root is added
    # by the CLI, never by editing the archived source package.
    from fedxmag.pcap_conn import SYN, ConnTracker, _frame, _resolutions

    class HTTPTracker(ConnTracker):
        def __init__(self) -> None:
            super().__init__()
            self.streams = {}
            self.extras = {}
            self.packet_seq = 0
            self.packet_flags = 0
            self.frame_end = 0

        def tcp(self, now, a, b, flags, seq, length, data, offset):
            self.packet_seq, self.packet_flags = seq, flags
            super().tcp(now, a, b, flags, seq, length, data, offset)

        def _payload(self, conn, side, length, data, offset):
            super()._payload(conn, side, length, data, offset)
            if conn.proto != "tcp" or length <= 0:
                return
            first = bytes(data[offset : min(offset + length, self.frame_end)])
            eligible = conn.resp[1] in {80, 81, 8000, 8080, 8888} or first.startswith(
                (b"GET ", b"POST ", b"HEAD ", b"HTTP/")
            )
            if id(conn) not in self.streams:
                if not eligible:
                    return
                self.streams[id(conn)] = ([bytearray(), bytearray()], [{}, {}], False)
            buffers, pending, clipped = self.streams[id(conn)]
            relative = (
                (self.packet_seq + bool(self.packet_flags & SYN)) - conn.base[side]
            ) & 0xFFFFFFFF
            if relative >= 16384:
                self.streams[id(conn)] = (buffers, pending, True)
                return
            payload = first[: 16384 - relative]
            if relative <= len(buffers[side]):
                overlap = len(buffers[side]) - relative
                buffers[side].extend(payload[overlap:])
            elif len(pending[side]) < 64:
                pending[side][relative] = payload
            for start in sorted(list(pending[side])):
                if start > len(buffers[side]):
                    break
                payload = pending[side].pop(start)
                buffers[side].extend(payload[len(buffers[side]) - start :])
            self.streams[id(conn)] = (buffers, pending, clipped or relative + length > 16384)

        def _finish(self, key, conn):
            stream = self.streams.pop(id(conn), None)
            super()._finish(key, conn)
            if stream is not None:
                buffers, _, clipped = stream
                vector = describe_http(bytes(buffers[0]), bytes(buffers[1]), clipped)
                if vector[0]:
                    self.extras[self.records[-1]] = vector

    tracker = HTTPTracker()
    names = {}

    def address(raw):
        if raw not in names:
            names[raw] = str(ipaddress.ip_address(raw))
        return names[raw]

    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        position, interfaces = 0, []
        while position + 12 <= len(data):
            kind, length = struct.unpack_from("<II", data, position)
            if length < 12 or position + length > len(data):
                raise ValueError("Truncated pcapng block")
            if kind == 0x0A0D0D0A:
                if struct.unpack_from("<I", data, position + 8)[0] != 0x1A2B3C4D:
                    raise ValueError("Unsupported pcapng endianness")
                interfaces = []
            elif kind == 1:
                interfaces.append(
                    (
                        struct.unpack_from("<H", data, position + 8)[0],
                        _resolutions(data, position, length),
                    )
                )
            elif kind == 6:
                interface, high, low, captured = struct.unpack_from("<IIII", data, position + 8)
                link, scale = interfaces[interface]
                tracker.counts["packets"] += 1
                if progress is not None and tracker.counts["packets"] % 1000000 == 0:
                    progress(position, len(data), tracker.counts["packets"])
                if link == 1:
                    tracker.frame_end = position + 28 + captured
                    _frame(
                        tracker,
                        tracker.counts,
                        data,
                        position + 28,
                        captured,
                        ((high << 32) | low) * scale,
                        address,
                        struct.unpack_from,
                    )
            position += length
    return tracker.close(), tracker.extras, dict(tracker.counts)


def run(args: argparse.Namespace) -> None:
    import sys

    import pandas as pd

    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    sys.path.insert(0, str(root / "src"))
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.data).resolve()
    with np.load(source) as payload:
        data = {key: payload[key] for key in payload.files}
    wanted = {int(value): index for index, value in enumerate(data["signature"])}
    extras = np.zeros((len(wanted), len(HTTP_FEATURES)), dtype=np.float32)
    matches = np.zeros(len(wanted), dtype=bool)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "status": "label_free_http_information_recovery",
            "input_sha256": sha256(source),
            "source_tracker_sha256": sha256(root / "src/fedxmag/pcap_conn.py"),
            "payload_limit_bytes_per_direction": 16384,
            "label_use": "none during extraction/join",
            "schema": list(HTTP_FEATURES),
        },
    )
    from fedxmag.pcap_conn import FIELDS

    reports = {}
    for day in ("Monday", "Thursday", "Friday"):
        path = root / "data/CIC-IDS2017" / f"{day}-WorkingHours.pcap"
        digest = hashlib.md5()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(block)
        published = path.with_suffix(".md5")
        if published.exists() and digest.hexdigest() != published.read_text().split()[0].lower():
            raise ValueError(f"Capture MD5 mismatch: {day}")
        print(json.dumps({"phase": "pcap_http", "day": day}), flush=True)
        records, observed, counts = extract(path)
        matched, http = 0, 0
        for start in range(0, len(records), 50000):
            block = records[start : start + 50000]
            strings = [
                [
                    str(r[0] // 10**9),
                    str(r[1]),
                    str(r[2]),
                    str(r[3]),
                    str(r[4]),
                    r[5],
                    r[6],
                    f"{r[7] / 10**9:.6f}",
                    str(r[8]),
                    str(r[9]),
                    r[10],
                    str(r[11]),
                    str(r[12]),
                ]
                for r in block
            ]
            frame = pd.DataFrame(strings, columns=FIELDS)
            for name in ("src_ip_bytes", "dst_ip_bytes", "missed_bytes", "http_uri"):
                frame[name] = "-"
            signatures = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
            for record, signature in zip(block, signatures, strict=True):
                index = wanted.get(int(signature))
                if index is not None:
                    matches[index] = True
                    matched += 1
                    if record in observed:
                        extras[index] = observed[record]
                        http += 1
        reports[day] = {
            "md5": digest.hexdigest(),
            "connections": len(records),
            "matched_sampled": matched,
            "http_observed_in_sample": http,
            "counts": counts,
        }
        write_json(output / f"{day}_report.json", reports[day])
        print(json.dumps({"day": day, "matched": matched, "http": http}), flush=True)
    if not matches.all():
        raise ValueError(f"HTTP join failed for {int((~matches).sum())} sampled connections")
    position = {str(name): index for index, name in enumerate(data["feature_names"])}
    data["features"][:, position["log_http_uri_length"]] = extras[:, 2]
    data["features"][:, position["uri_percent_fraction"]] = extras[:, 3]
    data["features"][:, position["uri_punctuation_fraction"]] = extras[:, 4]
    data["features"][:, position["available_http_uri"]] = extras[:, 1] > 0
    data["http_features"] = extras
    data["http_feature_names"] = np.asarray(HTTP_FEATURES)
    np.savez_compressed(output / "records.npz", **data)
    write_json(
        output / "report.json",
        {
            "status": "joined_every_sampled_record",
            "days": reports,
            "input_records": len(matches),
            "output_sha256": sha256(output / "records.npz"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
