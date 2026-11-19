"""EXPLORATORY DIAGNOSIS (after the sealed M1 test): raw-record view of X-IIoTID.

Reconstructs, with exactly the preprocessing rules (exact-row deduplication,
timestamp parsing, endpoint validation, per-date chronological window split),
which released records fall into which 60 s window and data role. Label-based
statistics are computed only for the already-consumed test band and for
normal-only reference rows of the train band. The audit band (60-70% of each
date), which no detector has ever scored, is never inspected so that it can
serve as an untouched confirmation source later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from fedxmag.m1.detection import auroc
from fedxmag.preprocessing.events import _clean, _parse_label, _timestamp_ns
from fedxmag.preprocessing.graph_builder import _split_ordered_items

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "select", "audit", "calibration", "test")
FRACTIONS = (0.50, 0.10, 0.10, 0.15, 0.15)
WINDOW_NS = 60 * 1_000_000_000
IDENTITY_COLUMNS = {"Date", "Timestamp", "Scr_IP", "Des_IP", "class1", "class2", "class3"}
CATEGORICAL_COLUMNS = {"Protocol", "Service"}
COLUMN_GROUPS = {
    "used_by_v3b_schema": ["Scr_bytes", "Des_bytes", "Scr_pkts", "Des_pkts"],
    "flow_unused": [
        "Scr_port", "Des_port", "Duration", "Conn_state", "missed_bytes", "is_syn_only",
        "Is_SYN_ACK", "is_pure_ack", "is_with_payload", "FIN or RST", "Bad_checksum",
        "is_SYN_with_RST", "Scr_ip_bytes", "Des_ip_bytes", "total_bytes", "total_packet",
        "paket_rate", "byte_rate", "Scr_packts_ratio", "Des_pkts_ratio", "Scr_bytes_ratio",
        "Des_bytes_ratio",
    ],
    "host_telemetry": [
        "Avg_user_time", "Std_user_time", "Avg_nice_time", "Std_nice_time",
        "Avg_system_time", "Std_system_time", "Avg_iowait_time", "Std_iowait_time",
        "Avg_ideal_time", "Std_ideal_time", "Avg_tps", "Std_tps", "Avg_rtps", "Std_rtps",
        "Avg_wtps", "Std_wtps", "Avg_ldavg_1", "Std_ldavg_1", "Avg_kbmemused",
        "Std_kbmemused", "Avg_num_Proc/s", "Std_num_proc/s", "Avg_num_cswch/s",
        "std_num_cswch/s",
    ],
    "host_log_activity": [
        "Login_attempt", "Succesful_login", "File_activity", "Process_activity",
        "read_write_physical.process", "is_privileged",
    ],
    "detector_alerts_label_proxy": ["anomaly_alert", "OSSEC_alert", "OSSEC_alert_level"],
}


def _number(value: str | None) -> float:
    cleaned = _clean(value)
    if cleaned is None:
        return math.nan
    lowered = cleaned.casefold()
    if lowered in {"true", "false"}:
        return 1.0 if lowered == "true" else 0.0
    try:
        return float(cleaned)
    except ValueError:
        return math.nan


def read_rows(path: Path) -> dict[str, object]:
    numeric = [column for group in COLUMN_GROUPS.values() for column in group]
    session_namespace = f"file-0000:{path.name}"
    seen: set[bytes] = set()
    sessions: list[str] = []
    timestamps: list[int] = []
    sources: list[str] = []
    destinations: list[str] = []
    labels: list[bool] = []
    categories: list[str] = []
    protocols: list[str] = []
    services: list[str] = []
    values: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        for row in reader:
            digest = hashlib.blake2b(
                "\x1f".join((row.get(field) or "") for field in fields).encode("utf-8"),
                digest_size=16,
            ).digest()
            if digest in seen:
                continue
            seen.add(digest)
            try:
                time_ns = _timestamp_ns(row.get("Timestamp"), row.get("Date"))
            except ValueError:
                continue
            source, destination = _clean(row.get("Scr_IP")), _clean(row.get("Des_IP"))
            if source is None or destination is None or source == destination:
                continue
            attack = _parse_label(row.get("class3"))
            date = _clean(row.get("Date"))
            sessions.append(f"{session_namespace}:{date}" if date else session_namespace)
            timestamps.append(time_ns)
            sources.append(source)
            destinations.append(destination)
            labels.append(attack)
            categories.append((_clean(row.get("class2")) or "") if attack else "")
            protocols.append(_clean(row.get("Protocol")) or "<unavailable>")
            services.append(_clean(row.get("Service")) or "<unavailable>")
            values.append([_number(row.get(column)) for column in numeric])
    return {
        "sessions": sessions,
        "timestamps": np.asarray(timestamps, dtype=np.int64),
        "sources": sources,
        "destinations": destinations,
        "labels": np.asarray(labels, dtype=bool),
        "categories": categories,
        "protocols": protocols,
        "services": services,
        "numeric_columns": numeric,
        "values": np.asarray(values, dtype=np.float64),
    }


def assign_bands(sessions: list[str], timestamps: np.ndarray) -> np.ndarray:
    """Replicates chronological_window_aligned_split (within_session_chronological)."""

    window_ids = timestamps // WINDOW_NS
    by_session: dict[str, set[int]] = defaultdict(set)
    for session, window in zip(sessions, window_ids.tolist(), strict=True):
        by_session[session].add(window)
    assignment: dict[tuple[str, int], int] = {}
    for session, windows in by_session.items():
        groups = _split_ordered_items(sorted(windows), FRACTIONS)
        for band, group in enumerate(groups):
            for window in group:
                assignment[(session, window)] = band
    return np.asarray(
        [assignment[(s, w)] for s, w in zip(sessions, window_ids.tolist(), strict=True)]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=PROJECT_ROOT / "data" / "X-IIoTID" / "X-IIoTID dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "diagnosis")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.input)
    sessions = rows["sessions"]
    bands = assign_bands(sessions, rows["timestamps"])
    windows = rows["timestamps"] // WINDOW_NS
    labels = rows["labels"]
    test = bands == SPLITS.index("test")
    train = bands == SPLITS.index("train")

    # Window table for the test band only (the consumed partition).
    window_records: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in np.flatnonzero(test):
        window_records[(sessions[index], int(windows[index]))].append(int(index))
    window_rows = []

    def endpoints_of(indices: list[int]) -> set[str]:
        return {rows["sources"][i] for i in indices} | {rows["destinations"][i] for i in indices}

    for (session, window), indices in sorted(window_records.items()):
        attack_rows = [i for i in indices if labels[i]]
        endpoints = endpoints_of(indices)
        attack_endpoints = endpoints_of(attack_rows)
        pairs = {(rows["sources"][i], rows["destinations"][i]) for i in indices}
        attack_pairs = {(rows["sources"][i], rows["destinations"][i]) for i in attack_rows}
        window_rows.append(
            {
                "session": session,
                "window_start_ns": int(window * WINDOW_NS),
                "records": len(indices),
                "attack_records": len(attack_rows),
                "attack_fraction": len(attack_rows) / len(indices),
                "nodes": len(endpoints),
                "attack_nodes": len(attack_endpoints),
                "pairs": len(pairs),
                "attack_pairs": len(attack_pairs),
                "categories": sorted({rows["categories"][i] for i in attack_rows}),
                "attack_endpoint_keys": sorted(attack_endpoints),
                "attack_pair_keys": sorted(attack_pairs),
                "protocol_counts": dict(Counter(rows["protocols"][i] for i in indices)),
            }
        )
    attack_windows = [w for w in window_rows if w["attack_records"]]
    normal_windows = [w for w in window_rows if not w["attack_records"]]
    fractions = np.asarray([w["attack_fraction"] for w in attack_windows])

    def quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
        array = np.asarray(values, dtype=float)
        levels = ("0.1", "0.25", "0.5", "0.75", "0.9")
        return {q: float(np.quantile(array, float(q))) for q in levels}

    per_date = defaultdict(lambda: {"normal_windows": 0, "attack_windows": 0})
    for window in window_rows:
        key = "attack_windows" if window["attack_records"] else "normal_windows"
        per_date[window["session"].split(":")[-1]][key] += 1
    mixed_dates = [d for d, c in per_date.items() if c["normal_windows"] and c["attack_windows"]]
    normal_in_mixed = sum(per_date[d]["normal_windows"] for d in mixed_dates)
    attack_in_mixed = sum(per_date[d]["attack_windows"] for d in mixed_dates)

    # Record-level univariate separability in the test band.
    record_auroc: dict[str, dict[str, float]] = {}
    test_attack = test & labels
    test_normal_rows = test & ~labels
    for group, columns in COLUMN_GROUPS.items():
        for column in columns:
            position = rows["numeric_columns"].index(column)
            column_values = rows["values"][:, position]
            normal = column_values[test_normal_rows]
            attack = column_values[test_attack]
            normal = normal[np.isfinite(normal)]
            attack = attack[np.isfinite(attack)]
            if normal.size < 20 or attack.size < 20:
                continue
            value = auroc(normal, attack)
            record_auroc[column] = {
                "group": group,
                "auroc": value,
                "separation": abs(value - 0.5) * 2,
                "normal_median": float(np.median(normal)),
                "attack_median": float(np.median(attack)),
            }
    group_summary = {
        group: {
            "max_separation": max(
                (v["separation"] for v in record_auroc.values() if v["group"] == group),
                default=math.nan,
            ),
            "top": sorted(
                ((k, round(v["auroc"], 3)) for k, v in record_auroc.items() if v["group"] == group),
                key=lambda item: -abs(item[1] - 0.5),
            )[:5],
        }
        for group in COLUMN_GROUPS
    }

    # Endpoint and protocol overlap: do attack windows live on the normal hosts?
    train_normal_endpoints = Counter()
    for index in np.flatnonzero(train & ~labels):
        train_normal_endpoints[rows["sources"][index]] += 1
        train_normal_endpoints[rows["destinations"][index]] += 1
    test_attack_endpoints = Counter()
    test_normal_endpoints = Counter()
    for index in np.flatnonzero(test):
        target = test_attack_endpoints if labels[index] else test_normal_endpoints
        target[rows["sources"][index]] += 1
        target[rows["destinations"][index]] += 1

    def seen_fraction(counter: Counter) -> float:
        total = sum(counter.values())
        return sum(c for e, c in counter.items() if e in train_normal_endpoints) / total

    def distribution(mask: np.ndarray, key: str) -> dict[str, float]:
        counts = Counter(rows[key][i] for i in np.flatnonzero(mask))
        total = sum(counts.values())
        return {k: v / total for k, v in counts.most_common(8)}

    report = {
        "status": "exploratory_post_test_diagnosis",
        "sealed": "audit band never inspected",
        "accepted_rows": int(len(sessions)),
        "band_rows": {name: int(np.sum(bands == i)) for i, name in enumerate(SPLITS)},
        "test_band": {
            "windows": len(window_rows),
            "normal_windows": len(normal_windows),
            "attack_windows": len(attack_windows),
            "attack_fraction_in_attack_windows": quantiles(fractions),
            "attack_windows_with_fraction_below_0.5": float(np.mean(fractions < 0.5)),
            "attack_windows_with_fraction_below_0.1": float(np.mean(fractions < 0.1)),
            "attack_records_per_attack_window": quantiles(
                [w["attack_records"] for w in attack_windows]
            ),
            "attack_node_share_in_attack_windows": quantiles(
                [w["attack_nodes"] / w["nodes"] for w in attack_windows]
            ),
            "attack_pair_share_in_attack_windows": quantiles(
                [w["attack_pairs"] / w["pairs"] for w in attack_windows]
            ),
            "dates": len(per_date),
            "dates_with_both_classes": len(mixed_dates),
            "normal_windows_on_mixed_dates": normal_in_mixed,
            "attack_windows_on_mixed_dates": attack_in_mixed,
            "per_date": dict(sorted(per_date.items())),
        },
        "record_level_univariate_auroc": record_auroc,
        "record_level_group_summary": group_summary,
        "endpoint_overlap": {
            "train_normal_endpoints": len(train_normal_endpoints),
            "test_normal_endpoints_seen_in_train_normal": seen_fraction(test_normal_endpoints),
            "test_attack_endpoints_seen_in_train_normal": seen_fraction(test_attack_endpoints),
        },
        "protocol_mix": {
            "train_normal": distribution(train & ~labels, "protocols"),
            "test_normal": distribution(test_normal_rows, "protocols"),
            "test_attack": distribution(test_attack, "protocols"),
        },
        "service_mix": {
            "train_normal": distribution(train & ~labels, "services"),
            "test_normal": distribution(test_normal_rows, "services"),
            "test_attack": distribution(test_attack, "services"),
        },
    }
    (args.output_dir / "raw_record_diagnosis.json").write_text(json.dumps(report, indent=2))
    with (args.output_dir / "test_band_windows.json").open("w", encoding="utf-8") as stream:
        json.dump(window_rows, stream)
    summary = {k: v for k, v in report["test_band"].items() if k != "per_date"}
    print(json.dumps(summary, indent=1))
    print(json.dumps(group_summary, indent=1))
    print(json.dumps(report["endpoint_overlap"], indent=1))
    print(json.dumps(report["protocol_mix"], indent=1))
    print(json.dumps(report["service_mix"], indent=1))


if __name__ == "__main__":
    main()
