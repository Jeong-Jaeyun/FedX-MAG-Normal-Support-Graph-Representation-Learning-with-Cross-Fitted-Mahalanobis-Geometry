"""Census of the TON_IoT processed network tables before any detector output exists.

`layout` reads only timestamps, endpoints, protocol, service, and connection state;
the `label` and `type` columns are never loaded. It informed the capture rule and the
adapter mapping. `support` then counts records and 60 s windows per capture by label,
under the ingestion rules of the frozen X-IIoTID reader. Neither phase computes a
feature distribution by label or any detector score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fedxmag.confirmation import sha256_file
from fedxmag.preprocessing.events import _clean

DATA = Path("data/TON-IoT/Processed_datasets/Processed_Network_dataset")
OUTPUT = Path("output/ton_iot_census")
COLUMNS = ["ts", "src_ip", "dst_ip", "proto", "service", "conn_state"]
GAP_SECONDS = 10 * 60
# Fixed after `layout`, before `support`: a capture is a testbed calendar day in AEST
# (UTC+10, no DST; Canberra DST ended 2019-04-07, before every dense day). Local
# midnight (14:00 UTC) falls in or near the nightly gaps the layout census showed.
CAPTURE_OFFSET_SECONDS = 10 * 3600
WINDOW_SECONDS = 60
MIN_SUPPORTED_NORMAL_WINDOWS = 10


def tables() -> list[Path]:
    paths = sorted(DATA.glob("Network_dataset_*.csv"), key=lambda p: int(p.stem.split("_")[-1]))
    if len(paths) != 23:
        raise FileNotFoundError(f"Expected 23 network tables under {DATA}, found {len(paths)}")
    return paths


def iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, UTC).isoformat()


def layout() -> dict[str, object]:
    files = []
    minute_records: Counter[int] = Counter()
    endpoints_by_day: dict[str, set[str]] = {}
    values = {column: Counter() for column in ("proto", "service", "conn_state")}
    fractional = 0
    for path in tables():
        frame = pd.read_csv(path, usecols=COLUMNS, dtype=str, keep_default_na=False)
        ts = pd.to_numeric(frame["ts"], errors="coerce")
        bad = int(ts.isna().sum())
        valid = ts.dropna().to_numpy(dtype=np.float64)
        fractional += int(np.count_nonzero(valid != np.floor(valid)))
        seconds = valid.astype(np.int64)
        minute_records.update(Counter((seconds // 60).tolist()))
        days = pd.to_datetime(seconds, unit="s", utc=True).strftime("%Y-%m-%d")
        kept = frame.loc[ts.notna()]
        for day, group in kept.groupby(np.asarray(days)):
            endpoints_by_day.setdefault(day, set()).update(group["src_ip"], group["dst_ip"])
        for column, counter in values.items():
            counter.update(frame[column].tolist())
        files.append(
            {
                "file": path.name,
                "sha256": sha256_file(path),
                "rows": len(frame),
                "bad_ts": bad,
                "first_ts": iso(float(valid.min())),
                "last_ts": iso(float(valid.max())),
                "backward_steps": int(np.count_nonzero(np.diff(seconds) < 0)),
            }
        )
        print(files[-1], flush=True)

    minutes = np.asarray(sorted(minute_records), dtype=np.int64)
    counts = np.asarray([minute_records[m] for m in minutes], dtype=np.int64)
    step = np.diff(minutes) * 60
    breaks = np.flatnonzero(step >= GAP_SECONDS)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(minutes) - 1]])
    segments = [
        {
            "start": iso(int(minutes[a]) * 60),
            "end": iso(int(minutes[b]) * 60 + 60),
            "active_minutes": int(b - a + 1),
            "records": int(counts[a : b + 1].sum()),
        }
        for a, b in zip(starts, ends, strict=True)
    ]
    day_of_minute = pd.to_datetime(minutes * 60, unit="s", utc=True).strftime("%Y-%m-%d")
    days = {}
    for day in sorted(set(day_of_minute)):
        mask = np.asarray(day_of_minute) == day
        days[day] = {
            "active_minutes": int(mask.sum()),
            "records": int(counts[mask].sum()),
            "max_records_per_minute": int(counts[mask].max()),
            "endpoints": len(endpoints_by_day[day]),
        }
    return {
        "columns_read": COLUMNS,
        "label_columns_read": False,
        "files": files,
        "rows": sum(item["rows"] for item in files),
        "fractional_timestamps": fractional,
        "active_minutes": len(minutes),
        "records_per_active_minute_quantiles": {
            str(q): float(np.quantile(counts, q)) for q in (0.5, 0.9, 0.99, 1.0)
        },
        "gap_seconds_for_segments": GAP_SECONDS,
        "segments": segments,
        "utc_days": days,
        "values": {name: dict(counter.most_common()) for name, counter in values.items()},
    }


def capture_of(seconds: int) -> str:
    return datetime.fromtimestamp(seconds + CAPTURE_OFFSET_SECONDS, UTC).strftime("%Y-%m-%d")


def support() -> dict[str, object]:
    """Record and window counts per capture, after the frozen reader's row rules."""

    seen: set[bytes] = set()
    rows = Counter()
    label_type = Counter()
    # (capture, minute) -> [normal records, attack records, attack types]
    windows: dict[tuple[str, int], list] = defaultdict(lambda: [0, 0, set()])
    for path in tables():
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            fields = next(reader)
            index = {
                name: fields.index(name) for name in ("ts", "src_ip", "dst_ip", "label", "type")
            }
            width = len(fields)
            for row in reader:
                rows["total"] += 1
                values = (row + [""] * width)[:width]
                digest = hashlib.blake2b(
                    "\x1f".join(values).encode("utf-8"), digest_size=16
                ).digest()
                if digest in seen:
                    rows["exact_duplicate"] += 1
                    continue
                seen.add(digest)
                try:
                    seconds = int(float(values[index["ts"]]))
                except ValueError:
                    rows["bad_timestamp"] += 1
                    continue
                source = _clean(values[index["src_ip"]])
                destination = _clean(values[index["dst_ip"]])
                if source is None or destination is None:
                    rows["missing_endpoint"] += 1
                    continue
                if source == destination:
                    rows["self_relation"] += 1
                    continue
                rows["kept"] += 1
                label, kind = values[index["label"]].strip(), values[index["type"]].strip()
                label_type[f"{label}|{kind}"] += 1
                window = windows[(capture_of(seconds), seconds // WINDOW_SECONDS)]
                if label == "0":
                    window[0] += 1
                else:
                    window[1] += 1
                    window[2].add(kind)
        print(path.name, dict(rows), flush=True)

    from fedxmag.preprocessing.graph_builder import _split_ordered_items

    fractions = [0.50, 0.10, 0.10, 0.15, 0.15]
    names = ["train", "select", "audit", "calibration", "test"]
    captures = {}
    for capture in sorted({key[0] for key in windows}):
        minutes = sorted(minute for key, minute in windows if key == capture)
        counts = [windows[(capture, minute)] for minute in minutes]
        normal = [i for i, item in enumerate(counts) if item[1] == 0]
        attack = [i for i, item in enumerate(counts) if item[1] > 0]
        types = Counter(kind for item in counts for kind in item[2])
        partitions = {}
        if len(minutes) >= len(fractions):
            groups = _split_ordered_items(list(range(len(minutes))), fractions)
            for name, group in zip(names, groups, strict=True):
                members = set(group)
                partitions[name] = {
                    "normal_windows": sum(i in members for i in normal),
                    "attack_windows": sum(i in members for i in attack),
                }
        commissioning = None
        if len(normal) >= MIN_SUPPORTED_NORMAL_WINDOWS:
            # Same rule as scripts/check_normal_commissioning.py.
            count = len(normal) // 5
            cutoff = normal[2 * count - 1]
            commissioning = {
                "reference_windows": count,
                "calibration_windows": count,
                "evaluation_normal_windows": len(normal) - 2 * count,
                "evaluation_attack_windows": sum(i > cutoff for i in attack),
                "earlier_attack_windows_excluded": sum(i <= cutoff for i in attack),
            }
        captures[capture] = {
            "first_window_utc": iso(minutes[0] * WINDOW_SECONDS),
            "last_window_utc": iso(minutes[-1] * WINDOW_SECONDS),
            "windows": len(minutes),
            "normal_windows": len(normal),
            "attack_windows": len(attack),
            "mixed_windows": sum(item[0] > 0 and item[1] > 0 for item in counts),
            "normal_records": sum(item[0] for item in counts),
            "attack_records": sum(item[1] for item in counts),
            "attack_type_windows": dict(types.most_common()),
            "supported_normal": len(normal) >= MIN_SUPPORTED_NORMAL_WINDOWS,
            "within_capture_partitions": partitions,
            "commissioning_20_20": commissioning,
        }
    return {
        "capture_rule": "AEST calendar day (UTC+10 fixed offset)",
        "window_seconds": WINDOW_SECONDS,
        "window_label": "attack if any kept record has label 1",
        "min_supported_normal_windows": MIN_SUPPORTED_NORMAL_WINDOWS,
        "row_rules": "exact-row dedup (blake2b over all fields), bad timestamp, "
        "missing endpoint, self relation; as fedxmag.preprocessing.events",
        "rows": dict(rows),
        "label_type": dict(label_type.most_common()),
        "captures": captures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["layout", "support"])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"{args.phase}.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    if args.phase == "support" and not (OUTPUT / "layout.json").exists():
        raise FileNotFoundError("Run the label-free layout census first")
    body = layout() if args.phase == "layout" else support()
    report = {"created_utc": datetime.now(UTC).isoformat(), **body}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
