"""Strictly past outgoing protocol-event context; no labels or identity values.

All raw events contribute to anonymous per-source counters. Query rows do not
update a same-second query. Address equality indexes histories only. Log counts
and fractions cover a fixed protocol taxonomy, not a hand-coded attack alarm.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .experiments import sha256, write_json
from .native_context import EVENT_DTYPE
from .protocol_fields import _numeric, _present, _text
from .record_experiment import corruption_groups

TAGS = (
    "icmp_unreachable",
    "icmp_redirect",
    "icmp_time_exceeded",
    "icmp_echo",
    "icmp_neighbor",
    "icmp_other",
    "dns_success",
    "dns_error",
    "tls_established",
    "tls_unestablished",
    "http_success",
    "http_error",
)
FLAG_COLUMNS = (
    "proto",
    "service",
    "src_port",
    "dns_query",
    "dns_rcode",
    "ssl_version",
    "ssl_established",
    "http_method",
    "http_status_code",
)
PROTOCOL_CONTEXT_FEATURES = tuple(
    f"context_outgoing_{seconds}s_{tag}_{statistic}"
    for seconds in (60, 300)
    for tag in TAGS
    for statistic in ("log_count", "fraction")
)


def event_flags(frame):
    proto, service = _text(frame, "proto").str.lower(), _text(frame, "service").str.lower()
    source, icmp = _numeric(frame, "src_port", proto.isin(("icmp", "icmp6", "ipv6-icmp")), 255)
    codes = ((3,), (5,), (11,), (0, 8, 128, 129), (133, 134, 135, 136, 143))
    flags = [icmp & np.isin(source, group) for group in codes]
    flags.append(icmp & ~np.isin(source, tuple(code for group in codes for code in group)))
    dns = service.str.contains("dns", regex=False) | _present(_text(frame, "dns_query"))
    rcode, available = _numeric(frame, "dns_rcode", dns, 65535)
    flags.extend((available & (rcode == 0), available & (rcode > 0)))
    tls = service.str.contains("ssl", regex=False) | _present(_text(frame, "ssl_version"))
    established = _text(frame, "ssl_established").str.lower()
    flags.extend(
        (
            tls.to_numpy() & established.isin(("t", "true", "1")).to_numpy(),
            tls.to_numpy() & established.isin(("f", "false", "0")).to_numpy(),
        )
    )
    http = service.str.contains("http", regex=False) | _present(_text(frame, "http_method"))
    status, available = _numeric(frame, "http_status_code", http, 999)
    flags.extend(
        (available & (status >= 200) & (status < 400), available & (status >= 400) & (status < 600))
    )
    return np.column_stack(flags).astype(np.uint8)


class RollingProtocols:
    """Small transparent reference for validating the native implementation."""

    def __init__(self, horizon, tags):
        self.horizon, self.tags = horizon, tags
        self.events = deque()
        self.nodes = {}

    def advance(self, now):
        while self.events and self.events[0][0] < now - self.horizon:
            _, key, flags = self.events.popleft()
            count, sums = self.nodes[key]
            if count == 1:
                del self.nodes[key]
            else:
                self.nodes[key] = (count - 1, sums - flags)

    def add(self, time, source, flags):
        count, sums = self.nodes.get(source, (0, np.zeros(self.tags, dtype=np.int64)))
        self.nodes[source] = (count + 1, sums + flags)
        self.events.append((time, source, flags.copy()))

    def query(self, source):
        result = np.zeros(2 * self.tags, dtype=np.float32)
        if source in self.nodes:
            count, sums = self.nodes[source]
            result[::2] = np.log1p(sums)
            result[1::2] = sums / count
        return result


def compile_library(destination):
    source = Path(__file__).resolve().parents[2] / "native/rolling_protocol_context.cpp"
    compiler = Path("D:/c/bin/g++.exe")
    subprocess.run(
        [
            str(compiler),
            "-std=c++17",
            "-O3",
            "-shared",
            "-static-libgcc",
            "-static-libstdc++",
            str(source),
            "-o",
            str(destination),
        ],
        check=True,
    )
    return source


def native_protocol_features(events, flags, queries, library):
    if events.dtype != EVENT_DTYPE or flags.dtype != np.uint8 or flags.ndim != 2:
        raise ValueError("Protocol native input ABI differs")
    if len(events) != len(flags) or not events.flags.c_contiguous or not flags.flags.c_contiguous:
        raise ValueError("Protocol flags and event rows must align contiguously")
    if np.any(events["time"][1:] < events["time"][:-1]) or np.any(flags > 1):
        raise ValueError("Protocol history requires chronological events and binary flags")
    output = np.zeros((queries, 4 * flags.shape[1]), dtype=np.float32)
    seen = np.zeros(queries, dtype=np.uint8)
    dll = ctypes.CDLL(str(library))
    function = dll.rolling_protocol_context
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    code = function(
        events.ctypes.data,
        flags.ctypes.data,
        len(events),
        flags.shape[1],
        queries,
        output.ctypes.data,
        seen.ctypes.data,
    )
    if code:
        raise ValueError(f"Native protocol context failed: {code}")
    if not seen.all():
        raise ValueError("Protocol context missed a query")
    return output


def pack_flags(task):
    filename, destination, expected = task
    flags = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.uint8, shape=(expected, len(TAGS))
    )
    offset = 0
    for frame in pd.read_csv(
        filename, usecols=list(FLAG_COLUMNS), dtype=str, keep_default_na=False, chunksize=200000
    ):
        values = event_flags(frame)
        flags[offset : offset + len(frame)] = values
        offset += len(frame)
    if offset != expected:
        raise ValueError("Raw protocol flag count differs from native event provenance")
    flags.flush()
    return {"file": Path(filename).name, "rows": offset, "flag_counts": flags.sum(0).tolist()}


def run(args):
    output, base, native = (
        Path(args.output).resolve(),
        Path(args.data).resolve(),
        Path(args.native_run).resolve(),
    )
    with np.load(base) as payload:
        data = {key: payload[key] for key in payload.files}
    if str(data["dataset"]) != "ton":
        raise ValueError("All-raw protocol flags currently require TON tables")
    if set(PROTOCOL_CONTEXT_FEATURES) & set(data["feature_names"]):
        raise ValueError("Protocol history is already present")
    registration = json.loads((native / "registration.json").read_text())
    with np.load(registration["query_data_path"]) as native_data:
        for key in ("signature", "observed_at", "timestamp"):
            if not np.array_equal(data[key], native_data[key]):
                raise ValueError("Protocol-history native query observations differ")
    raw_native = native
    visited = set()
    while True:
        if raw_native in visited:
            raise ValueError("Cyclic raw native provenance")
        visited.add(raw_native)
        original = json.loads((raw_native / "report.json").read_text())
        if "days" in original:
            break
        ancestor = json.loads((raw_native / "registration.json").read_text())
        raw_native = Path(ancestor["args"]["native_run"]).resolve()
    if any(part["invalid_timestamp"] for part in original["days"]):
        raise ValueError("Flag provenance requires time-valid TON raw rows")
    output.mkdir(parents=True, exist_ok=False)
    library = output / "rolling_protocol_context.dll"
    source = compile_library(library)
    (output / source.name).write_bytes(source.read_bytes())
    root = Path(args.root) / "data/TON-IoT/Processed_datasets/Processed_Network_dataset"
    paths = [root / part["file"] for part in original["days"]]
    source_hashes = {p.name: sha256(p) for p in paths}
    base_registration = json.loads((base.parent / "registration.json").read_text())
    if source_hashes != base_registration["raw_sources_sha256"]:
        raise ValueError("Protocol-event raw sources differ from current feature sources")
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": "ton",
            "input_sha256": sha256(base),
            "native_event_sha256": sha256(native / "events_sorted.npy"),
            "native_cpp_sha256": sha256(source),
            "native_library_sha256": sha256(library),
            "raw_source_sha256": source_hashes,
            "fit": "none; fixed outgoing protocol taxonomy; every raw event; strictly past seconds",
            "labels_used": "none",
            "query_observations": "all existing rows unchanged",
        },
    )
    packed = output / "packed"
    packed.mkdir()
    tasks = [
        (str(path), str(packed / f"flags_{index:02}.npy"), part["events"])
        for index, (path, part) in enumerate(zip(paths, original["days"], strict=True))
    ]
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for report in pool.map(pack_flags, tasks):
            reports.append(report)
            print(report, flush=True)
    events = np.load(native / "events_unsorted.npy", mmap_mode="r")
    order = np.argsort(events["time"], kind="stable")
    flags = np.lib.format.open_memmap(
        output / "flags_unsorted.npy", mode="w+", dtype=np.uint8, shape=(len(events), len(TAGS))
    )
    offset = 0
    for _, path, count in tasks:
        flags[offset : offset + count] = np.load(path, mmap_mode="r")
        offset += count
    flags.flush()
    chronological = np.load(native / "events_sorted.npy", mmap_mode="r")
    ordered = np.lib.format.open_memmap(
        output / "flags_sorted.npy", mode="w+", dtype=np.uint8, shape=flags.shape
    )
    for begin in range(0, len(flags), 500000):
        ordered[begin : begin + 500000] = flags[order[begin : begin + 500000]]
    ordered.flush()
    # Actual raw prefix audit uses a COPY of query slots, never changes input files.
    prefix = chronological[:10000].copy()
    queried = np.flatnonzero(prefix["query"] >= 0)
    prefix["query"] = -1
    prefix["query"][queried] = np.arange(len(queried))
    actual = native_protocol_features(prefix, np.asarray(ordered[:10000]), len(queried), library)
    expected = np.zeros_like(actual)
    histories = [RollingProtocols(h, len(TAGS)) for h in (60, 300)]
    start = 0
    while start < len(prefix):
        stop = start + 1
        while stop < len(prefix) and prefix["time"][stop] == prefix["time"][start]:
            stop += 1
        for history in histories:
            history.advance(int(prefix["time"][start]))
        for i in range(start, stop):
            if prefix["query"][i] >= 0:
                expected[prefix["query"][i]] = np.concatenate(
                    [h.query(int(prefix["source"][i])) for h in histories]
                )
        for i in range(start, stop):
            for history in histories:
                history.add(int(prefix["time"][i]), int(prefix["source"][i]), ordered[i])
        start = stop
    error = float(np.max(np.abs(actual - expected))) if len(actual) else 0.0
    if not np.allclose(actual, expected, rtol=1e-6, atol=1e-6):
        raise ValueError("Actual raw protocol prefix differs from Python history")
    extra = native_protocol_features(chronological, ordered, len(data["features"]), library)
    data["features"] = np.concatenate((data["features"], extra), axis=1)
    data["feature_names"] = np.r_[data["feature_names"], np.array(PROTOCOL_CONTEXT_FEATURES)]
    corruption_groups(data["feature_names"])
    np.savez_compressed(output / "records.npz", **data)
    with np.load(base) as old:
        if not np.array_equal(data["features"][:, : old["features"].shape[1]], old["features"]):
            raise ValueError("Protocol context changed existing features")
    write_json(
        output / "report.json",
        {
            "records": len(extra),
            "dimensions": data["features"].shape[1],
            "tags": list(TAGS),
            "raw_flag_parts": reports,
            "old_features_exact": True,
            "actual_prefix_query_rows": len(queried),
            "native_python_max_absolute_error": error,
            "output_sha256": sha256(output / "records.npz"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "native-run", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())
