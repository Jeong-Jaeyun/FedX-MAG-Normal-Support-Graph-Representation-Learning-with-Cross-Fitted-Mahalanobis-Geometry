"""Global time-order context preparation, including TON's 23 separate CSVs.

Compact memory-mapped events avoid retaining millions of Python event objects.
A local C++ implementation computes the same anonymous features as the Python
reference. No label/type column is read while packing or querying event history.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .experiments import sha256, write_json
from .record_data import NUMERIC, XIIOTID_COLUMNS
from .record_experiment import corruption_groups
from .temporal_context import CONTEXT_FEATURES, FAILED_STATES

EVENT_DTYPE = np.dtype(
    [
        ("time", "<i8"),
        ("source", "<u8"),
        ("destination", "<u8"),
        ("port", "<i8"),
        ("failed", "<f8"),
        ("bytes", "<f8"),
        ("packets", "<f8"),
        ("query", "<i8"),
    ]
)


def engine_path() -> Path:
    base = Path(__file__).resolve().parents[2]
    source = base / "native/rolling_context.cpp"
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("The local g++ compiler is required for native context extraction")
    cache = base / "artifacts/native_context_engine"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"context_{sha256(source)[:16]}.{'dll' if os.name == 'nt' else 'so'}"
    if not target.exists():
        flags = ["-static"] if os.name == "nt" else ["-fPIC"]
        subprocess.run(
            [
                compiler,
                "-std=c++17",
                "-O3",
                "-Wall",
                "-Wextra",
                "-shared",
                *flags,
                str(source),
                "-o",
                str(target),
            ],
            check=True,
        )
    return target


def native_features(events: np.ndarray, queries: int) -> tuple[np.ndarray, np.ndarray]:
    events = np.ascontiguousarray(events, dtype=EVENT_DTYPE)
    result = np.zeros((queries, len(CONTEXT_FEATURES)), dtype=np.float32)
    seen = np.zeros(queries, dtype=np.uint8)
    library = ctypes.CDLL(str(engine_path()))
    function = library.compute_context
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    status = function(
        events.ctypes.data, len(events), queries, result.ctypes.data, seen.ctypes.data
    )
    if status:
        raise ValueError(f"Native context failed ({status}): sort/bounds/allocation/alias check")
    if not np.isfinite(result).all():
        raise ValueError("Native context produced a nonfinite feature")
    return result, seen.astype(bool)


def pack_file(task: tuple[str, str, str]) -> dict:
    filename, input_path, destination = task
    with np.load(input_path) as payload:
        wanted = payload["signature"]
    order = np.argsort(wanted)
    sorted_wanted = wanted[order]
    fields = (
        "ts",
        "src_ip",
        "dst_ip",
        "proto",
        "service",
        "conn_state",
        "src_port",
        "dst_port",
        *NUMERIC,
        "http_uri",
    )
    blocks, rows, invalid = [], 0, 0
    for frame in pd.read_csv(
        filename,
        dtype=str,
        keep_default_na=False,
        chunksize=200000,
        usecols=lambda name: XIIOTID_COLUMNS.get(name, name) in fields,
    ):
        frame = frame.rename(columns=XIIOTID_COLUMNS)
        rows += len(frame)
        for name in fields:
            if name not in frame:
                frame[name] = "-"
        timestamp = pd.to_numeric(frame["ts"], errors="coerce").to_numpy()
        valid = np.isfinite(timestamp)
        invalid += int((~valid).sum())
        frame, timestamp = frame.loc[valid], timestamp[valid]
        if not len(frame):
            continue

        def numeric(name: str, observed_frame: pd.DataFrame = frame) -> np.ndarray:
            values = pd.to_numeric(observed_frame[name], errors="coerce").to_numpy(dtype=np.float64)
            return np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0)

        events = np.empty(len(frame), dtype=EVENT_DTYPE)
        events["time"] = np.ceil(timestamp + numeric("duration")).astype(np.int64)
        events["source"] = pd.util.hash_array(frame["src_ip"].to_numpy())
        events["destination"] = pd.util.hash_array(frame["dst_ip"].to_numpy())
        events["port"] = pd.to_numeric(frame["dst_port"], errors="coerce").fillna(-1).to_numpy()
        events["failed"] = frame["conn_state"].isin(FAILED_STATES).to_numpy()
        events["bytes"] = numeric("src_bytes") + numeric("dst_bytes")
        events["packets"] = numeric("src_pkts") + numeric("dst_pkts")
        signature = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
        positions = np.searchsorted(sorted_wanted, signature)
        found = (positions < len(wanted)) & (
            sorted_wanted[np.minimum(positions, len(wanted) - 1)] == signature
        )
        events["query"] = -1
        events["query"][found] = order[positions[found]]
        blocks.append(events)
    packed = np.concatenate(blocks)
    np.save(destination, packed)
    return {
        "file": Path(filename).name,
        "rows": rows,
        "invalid_timestamp": invalid,
        "events": len(packed),
        "packed": destination,
    }


def run(args: argparse.Namespace) -> None:
    root, source, output = (
        Path(args.root).resolve(),
        Path(args.data).resolve(),
        Path(args.output).resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    packed_directory = output / "packed"
    packed_directory.mkdir()
    with np.load(source) as payload:
        data = {key: payload[key] for key in payload.files}
    dataset = str(data.get("dataset", "ton"))
    if dataset == "ton":
        paths = sorted(
            (root / "data/TON-IoT/Processed_datasets/Processed_Network_dataset").glob(
                "Network_dataset_*.csv"
            ),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        if len(paths) != 23:
            raise ValueError("Expected all 23 TON connection tables")
    elif dataset == "xiiotid":
        paths = [root / "data/X-IIoTID/X-IIoTID dataset.csv"]
    elif dataset == "cic":
        paths = [
            root / f"data/processed/cic_ids2017_conn/{day}.csv"
            for day in ("Monday", "Thursday", "Friday")
        ]
    else:
        raise ValueError("Unknown dataset")
    library = engine_path()
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": dataset,
            "input_sha256": sha256(source),
            "engine_sha256": sha256(library),
            "native_source_sha256": sha256(
                Path(__file__).resolve().parents[2] / "native/rolling_context.cpp"
            ),
            "raw_files_sha256": {p.name: sha256(p) for p in paths},
            "history": "all time-valid unlabelled events, globally sorted across every file",
            "same_second": "all queries before any event in their second is admitted",
        },
    )
    tasks = [
        (str(path), str(source), str(packed_directory / f"part_{i:02}.npy"))
        for i, path in enumerate(paths)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = []
        for report in pool.map(pack_file, tasks):
            reports.append(report)
            print({key: report[key] for key in ("file", "rows", "events")}, flush=True)
    count = sum(report["events"] for report in reports)
    combined = np.lib.format.open_memmap(
        output / "events_unsorted.npy", mode="w+", dtype=EVENT_DTYPE, shape=(count,)
    )
    position = 0
    for report in reports:
        part = np.load(report["packed"], mmap_mode="r")
        combined[position : position + len(part)] = part
        position += len(part)
    combined.flush()
    order = np.argsort(combined["time"], kind="stable")
    ordered = np.lib.format.open_memmap(
        output / "events_sorted.npy", mode="w+", dtype=EVENT_DTYPE, shape=(count,)
    )
    for start in range(0, count, 500000):
        ordered[start : start + 500000] = combined[order[start : start + 500000]]
    ordered.flush()
    print({"phase": "global_native_context", "events": count}, flush=True)
    context, seen = native_features(ordered, len(data["signature"]))
    if not seen.all():
        raise ValueError(f"Native context did not join {int((~seen).sum())} prepared records")
    data["features"] = np.concatenate((data["features"], context), axis=1)
    data["feature_names"] = np.concatenate((data["feature_names"], np.asarray(CONTEXT_FEATURES)))
    data["dataset"] = np.asarray(dataset)
    corruption_groups(data["feature_names"])
    np.savez_compressed(output / "records.npz", **data)
    write_json(
        output / "report.json",
        {
            "events": count,
            "days": reports,
            "records": len(context),
            "dimensions": data["features"].shape[1],
            "output_sha256": sha256(output / "records.npz"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
