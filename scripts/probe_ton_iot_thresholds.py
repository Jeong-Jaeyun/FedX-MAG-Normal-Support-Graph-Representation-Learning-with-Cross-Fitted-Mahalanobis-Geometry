"""Post-hoc threshold probes behind the TON_IoT false-positive root cause (exploratory).

Reads output/ton_iot_fpr_diagnosis/samples.csv (from diagnose_ton_iot_fpr.py) and asks,
with the archived scores held fixed, which threshold rule restores the 5% target:
size-conditional (Mondrian) calibration, per-capture calibration, and calibration from
the most recent background samples with a delay. It compares the detector with the
record count under the same rules, overall and per attack type, and summarizes the
label-0 records of each commissioning phase for the four worst captures. TON_IoT is
consumed: every number here is exploratory and selects nothing.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from diagnose_ton_iot_fpr import DIAGNOSIS
from run_entity_detector import write_json
from run_ton_iot_validation import DATA, OUTPUT, _settings, tables

from fedxmag.flow_context import WINDOW_NS
from fedxmag.serialization import load_graph_samples
from fedxmag.ton_iot import capture_of, read_ton_records

SCORES = {"commissioned": "score", "recalibrated": "recal_score", "record_count": "record_score"}
FORENSIC_CAPTURES = ("2019-04-23", "2019-04-24", "2019-04-25", "2019-04-26")


def conformal(scores, target: float = 0.05) -> float:
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    rank = math.ceil((len(ordered) + 1) * (1 - target))
    return float(ordered[rank - 1]) if rank <= len(ordered) else math.nan


def rates(negative_alerts: np.ndarray, positive_alerts: np.ndarray,
          negative_captures: np.ndarray) -> dict[str, Any]:
    per = {str(c): float(negative_alerts[negative_captures == c].mean())
           for c in sorted(set(negative_captures))}
    return {"tpr": float(positive_alerts.mean()), "fpr": float(negative_alerts.mean()),
            "per_capture_fpr": per}


def main() -> None:
    destination = DIAGNOSIS / "threshold_probes.json"
    if destination.exists():
        raise FileExistsError(destination)
    frame = pd.read_csv(DIAGNOSIS / "samples.csv")
    frame["record_score"] = np.log1p(frame.records)
    calibration, negative, positive = (frame[frame.role == r] for r in
                                       ("calibration", "negative", "positive"))
    background = frame[frame.role != "positive"].sort_values(["capture", "minute"])
    caps = negative.capture.to_numpy()
    report: dict[str, Any] = {"status": "post_hoc_exploratory_on_consumed_ton_iot", "rules": {}}

    def static(name: str, score: str, threshold_of) -> None:
        report["rules"].setdefault(name, {})[score] = rates(
            negative[score].to_numpy() > threshold_of(negative),
            positive[score].to_numpy() > threshold_of(positive), caps)

    for score in SCORES.values():
        static("pooled", score, lambda part, s=score: conformal(calibration[s]))
        own = {c: conformal(p[score]) for c, p in calibration.groupby("capture")}
        static("per_capture", score, lambda part, own=own: part.capture.map(own).to_numpy())
        for size in ("records", "flows"):
            edges = np.quantile(calibration[size], [0.2, 0.4, 0.6, 0.8])
            bins = np.asarray([conformal(calibration[score][np.digitize(calibration[size], edges)
                                                            == b]) for b in range(5)])
            static(f"mondrian_{size}_x5", score,
                   lambda part, e=edges, b=bins, z=size: b[np.digitize(part[z], e)])
        for memory in (20, 40, 100):
            for delay in (0, 30, 120):
                alerts, covered = {}, {}
                for role, part in (("negative", negative), ("positive", positive)):
                    flags, ok = [], []
                    for capture, rows in part.groupby("capture", sort=False):
                        past = background[background.capture == capture]
                        minutes, values = past.minute.to_numpy(), past[score].to_numpy()
                        for minute, value in zip(rows.minute, rows[score], strict=True):
                            recent = values[minutes < minute - delay][-memory:]
                            supported = len(recent) >= memory
                            ok.append(supported)
                            flags.append(supported and value > conformal(recent))
                    alerts[role], covered[role] = np.asarray(flags), np.asarray(ok)
                neg_caps = np.concatenate([rows.capture.to_numpy() for _, rows in
                                           negative.groupby("capture", sort=False)])
                entry = rates(alerts["negative"][covered["negative"]],
                              alerts["positive"][covered["positive"]],
                              neg_caps[covered["negative"]])
                entry["covered"] = float(covered["negative"].mean())
                if memory == 20 and delay == 0:
                    positive_order = pd.concat([rows for _, rows in
                                                positive.groupby("capture", sort=False)])
                    positive_order = positive_order.assign(alert=alerts["positive"])
                    report.setdefault("_rolling20_alerts", {})[score] = positive_order
                report["rules"].setdefault(f"recent_m{memory}_delay{delay}", {})[score] = entry

    groups, _ = _settings()
    attack = load_graph_samples(OUTPUT / "views/attack_graphs.pt", groups)
    types = {g.metadata["window_start_ns"] // WINDOW_NS: g.metadata["attack_categories"]
             for g in attack}
    per_type: dict[str, Any] = {}
    rolling = report.pop("_rolling20_alerts")
    for score, part in rolling.items():
        exploded = part.assign(types=part.minute.map(types)).explode("types")
        for kind, rows in exploded.groupby("types"):
            item = per_type.setdefault(kind, {"windows": len(rows),
                                              "median_records": float(rows.records.median())})
            item[f"{score}_tpr_recent_m20"] = float(rows.alert.mean())
    report["per_attack_type_recent_m20"] = per_type

    records = read_ton_records(tables(DATA))
    minutes = records.timestamps // WINDOW_NS
    capture_of_minute = {int(m): capture_of(int(m) * WINDOW_NS) for m in np.unique(minutes)}
    endpoints = np.asarray(records.endpoints, dtype=object)
    forensic: dict[str, Any] = {}
    for capture in FORENSIC_CAPTURES:
        day = np.flatnonzero(np.fromiter((capture_of_minute[int(m)] == capture for m in minutes),
                                         bool, len(minutes)))
        normal = day[~records.attack[day]]
        attackers = set(endpoints[records.sources[day[records.attack[day]]]]) | set(
            endpoints[records.destinations[day[records.attack[day]]]])
        cal = calibration[calibration.capture == capture]
        neg = negative[negative.capture == capture]
        phases = {
            "reference": normal[minutes[normal] < cal.minute.min()],
            "calibration": normal[np.isin(minutes[normal], cal.minute)],
            "evaluation_true_negative": normal[np.isin(minutes[normal],
                                                       neg.minute[~neg.alert_commissioned])],
            "evaluation_false_positive": normal[np.isin(minutes[normal],
                                                        neg.minute[neg.alert_commissioned])],
        }
        forensic[capture] = {}
        for phase, rows in phases.items():
            windows = len(np.unique(minutes[rows]))
            if not windows:
                continue
            sources = endpoints[records.sources[rows]]
            destinations = endpoints[records.destinations[rows]]
            touching = np.fromiter((s in attackers or d in attackers
                                    for s, d in zip(sources, destinations, strict=True)),
                                   bool, len(rows))
            states = Counter(records.state_names[c] for c in records.states[rows])
            forensic[capture][phase] = {
                "windows": windows,
                "records_per_window": len(rows) / windows,
                "touching_attack_endpoints_of_day": float(touching.mean()),
                "top_sources": {k: v / len(rows) for k, v in Counter(sources).most_common(4)},
                "top_states": {str(k): v / len(rows) for k, v in states.most_common(4)},
            }
    report["label0_composition"] = forensic
    report["finished_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    write_json(destination, report)
    for name, entry in report["rules"].items():
        print(name, {s: (round(v["tpr"], 3), round(v["fpr"], 3)) for s, v in entry.items()})
    print(json.dumps(per_type, indent=1))


if __name__ == "__main__":
    Path(DIAGNOSIS).mkdir(parents=True, exist_ok=True)
    main()
