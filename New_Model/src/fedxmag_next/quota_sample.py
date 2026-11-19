"""Label-independent minute coverage quota on top of TON's fixed 1/32 sample.

Keep the original sample and add up to sixteen observations to sparse minutes.
Selection uses completed time, old sampling membership, and mixed raw row index;
no label/type value is accessed until the already chosen rows are encoded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .experiments import sha256, write_json
from .native_context import EVENT_DTYPE, native_features
from .record_data import run as prepare_records
from .record_experiment import corruption_groups
from .temporal_context import CONTEXT_FEATURES


def mixed_rows(rows: np.ndarray) -> np.ndarray:
    values = rows.astype(np.uint64) + np.uint64(17)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def run(args: argparse.Namespace) -> None:
    output, native = Path(args.output).resolve(), Path(args.native_run).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((native / "report.json").read_text())
    registration = json.loads((native / "registration.json").read_text())
    if registration["dataset"] != "ton" or any(r["invalid_timestamp"] for r in report["days"]):
        raise ValueError("Raw-position reuse currently requires time-valid TON tables")
    with np.load(args.base_data) as payload:
        old = {key: payload[key] for key in ("signature", "observed_at", "features")}
    if registration["input_sha256"] != sha256(Path(args.base_data)):
        raise ValueError("Native query membership does not belong to the base sample")
    events = np.load(native / "events_unsorted.npy", mmap_mode="r")
    ordered_events = np.load(native / "events_sorted.npy", mmap_mode="r")
    order = np.argsort(events["time"], kind="stable")
    sorted_minutes = ordered_events["time"] // 60
    if not np.array_equal(events["time"][order], ordered_events["time"]):
        raise ValueError("Native event chronological alignment changed")
    observed_minutes, observed_counts = np.unique(old["observed_at"] // 60, return_counts=True)
    counts = dict(zip(observed_minutes.tolist(), observed_counts.tolist(), strict=True))
    boundaries = np.flatnonzero(np.r_[True, np.diff(sorted_minutes) != 0])
    selected = []
    for start, stop in zip(boundaries, np.r_[boundaries[1:], len(events)], strict=True):
        minute = int(sorted_minutes[start])
        need = max(0, args.minimum_per_minute - counts.get(minute, 0))
        if need:
            segment = ordered_events[start:stop]
            eligible = segment["query"] < 0
            if args.exclude_self_connections:
                eligible &= segment["source"] != segment["destination"]
            candidates = np.flatnonzero(eligible) + start
            raw = order[candidates]
            selected.extend(raw[np.argsort(mixed_rows(raw))[:need]].tolist())
    extra = np.unique(np.asarray(selected, dtype=np.int64))
    file_ends = np.cumsum([part["events"] for part in report["days"]])
    file_starts = np.r_[0, file_ends[:-1]]
    file_ids = np.searchsorted(file_ends, extra, side="right")
    row_directory = output / "quota_rows"
    row_directory.mkdir()
    for index in range(len(file_ends)):
        rows = extra[file_ids == index] - file_starts[index]
        np.save(row_directory / f"rows_{index:02}.npy", rows)
    prepared = output / "prepared"
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": "ton",
            "base_input_sha256": sha256(Path(args.base_data)),
            "native_events_sha256": sha256(native / "events_unsorted.npy"),
            "query_data_path": str(prepared / "records.npz"),
            "sampling": "union of stable 1/32 rows and mixed-row minimum per completed minute",
            "label_use_in_selection": "none",
            "self_connections": "excluded" if args.exclude_self_connections else "retained",
            "additional_raw_rows": len(extra),
        },
    )
    print({"phase": "minute_quota_preparation", "additional_raw_rows": len(extra)}, flush=True)
    prepare_records(
        argparse.Namespace(
            root=args.root,
            output=str(prepared),
            workers=args.workers,
            denominator=32,
            dataset="ton",
            extra_row_directory=str(row_directory),
            keep_self_connections=not args.exclude_self_connections,
        )
    )
    with np.load(prepared / "records.npz") as payload:
        data = {key: payload[key] for key in payload.files}
    positions = np.searchsorted(data["signature"], old["signature"])
    if not np.array_equal(data["signature"][positions], old["signature"]):
        raise ValueError("Quota preparation lost an original sampled observation")
    if not np.array_equal(data["features"][positions], old["features"]):
        raise ValueError("Quota preparation changed original measured features")
    new_events = np.lib.format.open_memmap(
        output / "events_unsorted.npy", mode="w+", dtype=EVENT_DTYPE, shape=events.shape
    )
    for start in range(0, len(events), 500000):
        new_events[start : start + 500000] = events[start : start + 500000]
    new_events["query"] = -1
    rows = file_starts[data["source_file_index"]] + data["source_row_index"]
    if not np.array_equal(new_events["time"][rows], data["observed_at"]):
        raise ValueError("Prepared row provenance differs from native completion times")
    new_events["query"][rows] = np.arange(len(rows))
    new_events.flush()
    sorted_new = np.lib.format.open_memmap(
        output / "events_sorted.npy", mode="w+", dtype=EVENT_DTYPE, shape=events.shape
    )
    for start in range(0, len(events), 500000):
        sorted_new[start : start + 500000] = new_events[order[start : start + 500000]]
    sorted_new.flush()
    context, seen = native_features(sorted_new, len(rows))
    if not seen.all():
        raise ValueError("Quota context failed to join every prepared row")
    data["features"] = np.concatenate((data["features"], context), axis=1)
    data["feature_names"] = np.concatenate((data["feature_names"], np.asarray(CONTEXT_FEATURES)))
    corruption_groups(data["feature_names"])
    with np.load(native / "records.npz") as original_context:
        if not np.array_equal(data["features"][positions], original_context["features"]):
            raise ValueError("Adding query observations changed existing causal context features")
    np.savez_compressed(output / "records.npz", **data)
    minutes, numbers = np.unique(data["observed_at"] // 60, return_counts=True)
    write_json(
        output / "report.json",
        {
            "records": len(rows),
            "old_records": len(old["signature"]),
            "additional_raw_rows": len(extra),
            "observed_minutes": len(minutes),
            "old_observed_minutes": len(observed_minutes),
            "minimum_observed_rows_per_minute": int(numbers.min()),
            "original_common_features_preserved": True,
            "original_context_features_preserved": True,
            "output_sha256": sha256(output / "records.npz"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", required=True)
    parser.add_argument("--native-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--minimum-per-minute", type=int, default=16)
    parser.add_argument("--exclude-self-connections", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
