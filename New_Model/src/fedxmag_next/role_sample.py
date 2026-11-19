"""Preserve active directed node roles while retaining every existing query row.

For each minute, add one hash-selected network observation for each source with
no outgoing query and each destination with no incoming query. Selection reads
native event times/keys/query membership only; labels/types are unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .experiments import sha256, write_json
from .native_context import EVENT_DTYPE, native_features
from .quota_sample import mixed_rows
from .record_data import FEATURES
from .record_data import run as prepare_records
from .record_experiment import corruption_groups
from .temporal_context import CONTEXT_FEATURES


def missing_role_rows(events, raw_indices):
    eligible = events["source"] != events["destination"]
    events, raw = events[eligible], raw_indices[eligible]
    if not len(raw):
        return np.empty(0, dtype=np.int64)
    ranks = mixed_rows(raw)
    added = []
    for field in ("source", "destination"):
        order = np.lexsort((ranks, events[field]))
        first = order[np.r_[True, np.diff(events[field][order]) != 0]]
        known = events[field][events["query"] >= 0]
        missing = ~np.isin(events[field][first], known)
        added.extend(raw[first[missing]].tolist())
    return np.unique(np.asarray(added, dtype=np.int64))


def run(args):
    base, native, output = (
        Path(args.base_data).resolve(),
        Path(args.native_run).resolve(),
        Path(args.output).resolve(),
    )
    origin = json.loads((native / "registration.json").read_text())
    native_report = json.loads((native / "report.json").read_text())
    if native_report.get("output_sha256") != sha256(base):
        raise ValueError("Native query events do not bind the supplied prepared observations")
    if origin.get("self_connections") != "excluded":
        raise ValueError("This ablation requires matched network-only observation scope")
    original_native = Path(origin["args"]["native_run"])
    raw_report = json.loads((original_native / "report.json").read_text())
    if any(row["invalid_timestamp"] for row in raw_report["days"]):
        raise ValueError("Raw row provenance requires time-valid TON tables")
    with np.load(base) as payload:
        old = {key: payload[key] for key in payload.files}
    if tuple(old["feature_names"]) != FEATURES + CONTEXT_FEATURES:
        raise ValueError("Expected common TON features and full causal history")
    events = np.load(native / "events_unsorted.npy", mmap_mode="r")
    chronological = np.load(native / "events_sorted.npy", mmap_mode="r")
    order = np.argsort(events["time"], kind="stable")
    if not np.array_equal(events["time"][order], chronological["time"]):
        raise ValueError("Chronological native alignment differs")
    ends = np.cumsum([part["events"] for part in raw_report["days"]])
    file_starts = np.r_[0, ends[:-1]]
    previous = file_starts[old["source_file_index"]] + old["source_row_index"]
    if not np.array_equal(events["query"][previous], np.arange(len(previous))):
        raise ValueError("Native query membership differs from prepared row provenance")
    if not np.array_equal(events["time"][previous], old["observed_at"]):
        raise ValueError("Native event times differ from old observations")
    minutes = chronological["time"] // 60
    boundaries = np.flatnonzero(np.r_[True, np.diff(minutes) != 0])
    extra = []
    for start, stop in zip(boundaries, np.r_[boundaries[1:], len(events)], strict=True):
        extra.extend(missing_role_rows(chronological[start:stop], order[start:stop]).tolist())
    extra = np.unique(np.asarray(extra, dtype=np.int64))
    selected = np.union1d(previous, extra)
    output.mkdir(parents=True, exist_ok=False)
    prepared = output / "prepared"
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": "ton",
            "base_input_sha256": sha256(base),
            "native_events_sha256": sha256(native / "events_unsorted.npy"),
            "query_data_path": str(prepared / "records.npz"),
            "sampling": (
                "retain all old rows; one outgoing/incoming observation per active role-minute"
            ),
            "label_use_in_selection": "none; event selection schema contains no labels/types",
            "self_connections": "excluded",
            "additional_raw_rows": len(extra),
        },
    )
    row_directory = output / "role_rows"
    row_directory.mkdir()
    file_ids = np.searchsorted(ends, selected, side="right")
    for index in range(len(ends)):
        np.save(
            row_directory / f"rows_{index:02}.npy", selected[file_ids == index] - file_starts[index]
        )
    print({"phase": "directed_role_preparation", "extra_raw_rows": len(extra)}, flush=True)
    prepare_records(
        argparse.Namespace(
            root=args.root,
            output=str(prepared),
            workers=args.workers,
            denominator=32,
            dataset="ton",
            extra_row_directory=str(row_directory),
            keep_self_connections=False,
        )
    )
    with np.load(prepared / "records.npz") as payload:
        data = {key: payload[key] for key in payload.files}
    positions = np.searchsorted(data["signature"], old["signature"])
    if not np.array_equal(data["signature"][positions], old["signature"]):
        raise ValueError("Role supplementation lost old observations")
    if not np.array_equal(data["features"][positions], old["features"][:, : len(FEATURES)]):
        raise ValueError("Role supplementation changed old measured features")
    new = np.lib.format.open_memmap(
        output / "events_unsorted.npy", mode="w+", dtype=EVENT_DTYPE, shape=events.shape
    )
    for begin in range(0, len(events), 500000):
        new[begin : begin + 500000] = events[begin : begin + 500000]
    new["query"] = -1
    rows = file_starts[data["source_file_index"]] + data["source_row_index"]
    if not np.array_equal(new["time"][rows], data["observed_at"]):
        raise ValueError("Augmented raw provenance differs")
    new["query"][rows] = np.arange(len(rows))
    new.flush()
    sorted_new = np.lib.format.open_memmap(
        output / "events_sorted.npy", mode="w+", dtype=EVENT_DTYPE, shape=events.shape
    )
    for begin in range(0, len(events), 500000):
        sorted_new[begin : begin + 500000] = new[order[begin : begin + 500000]]
    sorted_new.flush()
    # Independent cardinality audit, rather than repeating the selection algorithm.
    full_role_count, observed_role_count = 0, 0
    for start, stop in zip(boundaries, np.r_[boundaries[1:], len(events)], strict=True):
        part = sorted_new[start:stop]
        part = part[part["source"] != part["destination"]]
        for field in ("source", "destination"):
            full = len(np.unique(part[field]))
            actual = len(np.unique(part[field][part["query"] >= 0]))
            if full != actual:
                raise ValueError("An active directed node role is still missing")
            full_role_count += full
            observed_role_count += actual
    context, seen = native_features(sorted_new, len(rows))
    if not seen.all():
        raise ValueError("Role context join missed observations")
    data["features"] = np.concatenate((data["features"], context), axis=1)
    data["feature_names"] = np.concatenate((data["feature_names"], np.asarray(CONTEXT_FEATURES)))
    if not np.array_equal(data["features"][positions], old["features"]):
        raise ValueError("Additional queries changed old past context features")
    corruption_groups(data["feature_names"])
    np.savez_compressed(output / "records.npz", **data)
    write_json(
        output / "report.json",
        {
            "records": len(rows),
            "old_records": len(previous),
            "additional_raw_rows": len(extra),
            "all_old_features_bit_exact": True,
            "full_directed_role_minutes": full_role_count,
            "observed_directed_role_minutes": observed_role_count,
            "output_sha256": sha256(output / "records.npz"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-data", "native-run", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())
