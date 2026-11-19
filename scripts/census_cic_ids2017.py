"""Census of the converted CIC-IDS2017 connection tables before any detector output exists.

`layout` reads the label-free tables: per day, rows, local time span, active minutes,
records per minute, value counts, and, per local hour, the records that touch an
address named in the published attack schedule (the NAT address 172.16.0.1, the
205.174.165.0/24 attacker network, and victim-LAN traffic of the infiltration host
192.168.10.8). It checks whether the schedule rule of fedxmag.cic_ids2017 is usable;
it led to removing unknown-status records (attack infrastructure outside the published
intervals). `label` writes the labelled tables by that rule. `support` reads them with
the frozen TON_IoT reader and counts records and 60 s windows per capture by label. No
phase computes a feature distribution by label or any detector score.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fedxmag.cic_ids2017 import CAPTURE_OFFSET_SECONDS, DAYS, capture_of, write_labelled
from fedxmag.confirmation import sha256_file
from fedxmag.flow_context import WINDOW_NS
from fedxmag.ton_iot import read_ton_records

ROOT = Path(__file__).resolve().parents[1]
CONN = ROOT / "data/processed/cic_ids2017_conn"
LABELLED = CONN / "labelled"
OUTPUT = ROOT / "output/cic_ids2017_census"
GAP_SECONDS = 10 * 60
VETTING_MINIMUM = 20  # vetted windows a recent-calibration threshold needs at 5%


def local(seconds: float) -> str:
    return datetime.fromtimestamp(seconds + CAPTURE_OFFSET_SECONDS, UTC).strftime(
        "%Y-%m-%d %H:%M:%S")


def layout() -> dict[str, object]:
    days = {}
    for day in DAYS:
        path = CONN / f"{day}.csv"
        manifest = json.loads((CONN / f"{day}_manifest.json").read_text())
        if sha256_file(path) != manifest["table_sha256"]:
            raise ValueError(f"{path.name} differs from its conversion manifest")
        frame = pd.read_csv(path, dtype={"src_ip": str, "dst_ip": str}, keep_default_na=False)
        seconds = frame.ts.to_numpy(dtype=np.int64)
        minutes = seconds // 60
        per_minute = pd.Series(minutes).value_counts().sort_index()
        steps = np.diff(per_minute.index.to_numpy()) * 60
        hour = pd.Series((seconds + CAPTURE_OFFSET_SECONDS) // 3600 % 24)
        nat = (frame.src_ip == "172.16.0.1") | (frame.dst_ip == "172.16.0.1")
        external = frame.src_ip.str.startswith("205.174.165.") | frame.dst_ip.str.startswith(
            "205.174.165.")
        vista_lan = ((frame.src_ip == "192.168.10.8") & frame.dst_ip.str.startswith("192.168.10.")
                     ) | ((frame.dst_ip == "192.168.10.8") & frame.src_ip.str.startswith(
                         "192.168.10."))
        by_hour = {
            name: {int(h): int(n) for h, n in hour[mask].value_counts().sort_index().items()}
            for name, mask in (("nat_172.16.0.1", nat), ("attacker_net_205.174.165", external),
                               ("vista_192.168.10.8_lan", vista_lan))
        }
        nat_peers = Counter(np.where(frame.src_ip[nat] == "172.16.0.1", frame.dst_ip[nat],
                                     frame.src_ip[nat]))
        external_hosts = Counter(np.concatenate([
            frame.src_ip[frame.src_ip.str.startswith("205.174.165.")].to_numpy(),
            frame.dst_ip[frame.dst_ip.str.startswith("205.174.165.")].to_numpy()]))
        days[day] = {
            "table_sha256": manifest["table_sha256"],
            "rows": len(frame),
            "first_local": local(seconds.min()),
            "last_local": local(seconds.max()),
            "active_minutes": len(per_minute),
            "minute_gaps_over_10_min": int(np.count_nonzero(steps >= GAP_SECONDS)),
            "records_per_minute_quantiles": {
                str(q): float(per_minute.quantile(q)) for q in (0.05, 0.5, 0.95, 1.0)},
            "records_per_local_hour": {int(h): int(n) for h, n in
                                       hour.value_counts().sort_index().items()},
            "endpoints": int(pd.concat([frame.src_ip, frame.dst_ip]).nunique()),
            "self_relation": int((frame.src_ip == frame.dst_ip).sum()),
            "values": {column: dict(Counter(frame[column]).most_common(12))
                       for column in ("proto", "service", "conn_state")},
            "schedule_address_records_by_local_hour": by_hour,
            "nat_top_peers": dict(nat_peers.most_common(8)),
            "attacker_net_hosts": dict(external_hosts.most_common(8)),
        }
        print(day, days[day]["rows"], days[day]["first_local"], days[day]["last_local"],
              flush=True)
    return {"label_columns_read": False, "capture_offset_seconds": CAPTURE_OFFSET_SECONDS,
            "days": days}


def label_tables() -> dict[str, object]:
    LABELLED.mkdir(parents=True, exist_ok=True)
    report = {}
    for day in DAYS:
        source, target = CONN / f"{day}.csv", LABELLED / f"{day}.csv"
        counts = write_labelled(source, target)
        # Monday is benign by the dataset definition; the rule must agree.
        if day == "Monday" and set(counts) - {"normal"}:
            raise ValueError(f"Monday has non-normal labels: {counts}")
        report[day] = {"source_sha256": sha256_file(source), "table": target.name,
                       "table_sha256": sha256_file(target), "rows_by_label": counts}
        print(day, counts, flush=True)
    return {"rule": "fedxmag.cic_ids2017.SCHEDULE", "code_sha256": sha256_file(
        ROOT / "src/fedxmag/cic_ids2017.py"), "tables": report}


def support() -> dict[str, object]:
    labelled = json.loads((OUTPUT / "label.json").read_text())
    paths = [LABELLED / f"{day}.csv" for day in DAYS]
    for day, path in zip(DAYS, paths, strict=True):
        if sha256_file(path) != labelled["tables"][day]["table_sha256"]:
            raise ValueError(f"{path.name} changed after labelling")
    records = read_ton_records(paths)
    windows = records.windows()
    captures: dict[str, dict[str, object]] = {}
    by_capture: dict[str, list[int]] = defaultdict(list)
    for minute in sorted(windows):
        by_capture[capture_of(minute * WINDOW_NS)].append(minute)
    for capture, minutes in by_capture.items():
        kinds, sizes, status = Counter(), {"natural_normal": [], "attack": []}, []
        for minute in minutes:
            rows = windows[minute]
            attack = records.attack[rows]
            kind = "attack" if attack.any() else "natural_normal"
            status.append(kind)
            sizes[kind].append(len(rows))
            if attack.any():
                kinds.update({records.type_names[t] for t in records.types[rows[attack]]})
        natural = [i for i, s in enumerate(status) if s == "natural_normal"]
        supported = [i for i, s in enumerate(status)
                     if sum(j < i for j in natural) >= VETTING_MINIMUM]
        captures[capture] = {
            "first_window_local": local(minutes[0] * 60),
            "last_window_local": local(minutes[-1] * 60),
            "windows": len(minutes),
            **{f"{kind}_windows": status.count(kind) for kind in ("natural_normal", "attack")},
            "attack_records": int(sum(records.attack[windows[m]].sum() for m in minutes)),
            "attack_type_windows": dict(kinds.most_common()),
            "records_per_window_median": {k: float(np.median(v)) if v else None
                                          for k, v in sizes.items()},
            f"windows_with_{VETTING_MINIMUM}_earlier_natural_normal": {
                kind: sum(status[i] == kind for i in supported)
                for kind in ("natural_normal", "attack")},
        }
        print(capture, captures[capture], flush=True)
    return {"ingestion": records.report, "window_label": "attack if any kept record has label 1",
            "captures": captures}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["layout", "label", "support"])
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"{args.phase}.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    body = {"layout": layout, "label": label_tables, "support": support}[args.phase]()
    report = {"created_utc": datetime.now(UTC).isoformat(timespec="seconds"), **body}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
