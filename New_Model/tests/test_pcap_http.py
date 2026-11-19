from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from fedxmag_next.pcap_http import describe_http, extract


def test_http_summary_uses_payload_structure_without_labels() -> None:
    request = b"GET /search?q=%3Ctag%3E HTTP/1.1\r\nHost: example.test\r\n\r\n"
    response = b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n"
    result = describe_http(request, response)
    assert result[0] == 1
    assert result[1] == np.float32(np.log1p(1))
    assert result[3] > 0 and result[5] > result[4]
    assert result[10] == 1 and result[12] == 1
    assert not describe_http(b"not an HTTP stream", b"").any()


def _tcp(
    source: bytes,
    destination: bytes,
    sport: int,
    dport: int,
    flags: int,
    sequence: int,
    payload: bytes = b"",
) -> bytes:
    tcp = struct.pack("!HHIIBBHHH", sport, dport, sequence, 0, 5 << 4, flags, 1024, 0, 0) + payload
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(tcp), 0, 0, 64, 6, 0, source, destination)
    return b"\0" * 12 + b"\x08\x00" + ip + tcp


def _block(kind: int, body: bytes) -> bytes:
    return struct.pack("<II", kind, 12 + len(body)) + body + struct.pack("<I", 12 + len(body))


def test_reassembly_handles_out_of_order_and_retransmission_without_changing_flows(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "src"))
    from fedxmag.pcap_conn import track_pcapng

    client, server = bytes([10, 0, 0, 1]), bytes([10, 0, 0, 2])
    request = b"GET /search?q=%3Ctag%3E HTTP/1.1\r\nHost: example.test\r\n\r\n"
    split = 19
    frames = [
        _tcp(client, server, 50000, 80, 2, 100),
        _tcp(server, client, 80, 50000, 18, 900),
        _tcp(client, server, 50000, 80, 24, 101 + split, request[split:]),
        _tcp(client, server, 50000, 80, 24, 101, request[:split]),
        _tcp(client, server, 50000, 80, 24, 101, request[:split]),
        _tcp(server, client, 80, 50000, 24, 901, b"HTTP/1.1 401 Unauthorized\r\n\r\n"),
    ]
    blob = _block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
    blob += _block(1, struct.pack("<HHI", 1, 0, 65535))
    for index, frame in enumerate(frames):
        body = struct.pack("<IIIII", 0, 0, 1000000 + index * 10000, len(frame), len(frame))
        blob += _block(6, body + frame + b"\0" * (-len(frame) % 4))
    path = tmp_path / "stream.pcapng"
    path.write_bytes(blob)
    original = track_pcapng(path).close()
    records, summaries, _ = extract(path)
    assert records == original
    result = summaries[records[0]]
    assert result[0] == 1 and result[1] == np.float32(np.log1p(1))
    assert result[2] == np.float32(np.log1p(len(b"/search?q=%3Ctag%3E")))
    assert result[12] == 1
