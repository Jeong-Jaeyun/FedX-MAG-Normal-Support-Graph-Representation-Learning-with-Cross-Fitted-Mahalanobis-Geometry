"""Exploratory method development on consumed data (X-IIoTID, TON_IoT); selects nothing final.

1. X-IIoTID: the archived raw-node + flow fusion scores against a record-count score
   (log1p of forward records per window), both thresholded on calibration-normal.
2. TON_IoT (scores fixed, from output/ton_iot_fpr_diagnosis/samples.csv): threshold
   policies for the commissioned score, the record count, and their maximum after
   calibration-q95 scaling, namely static pooled, recent vetted background samples
   (memory m, delay d; windows without enough earlier samples are left out and the
   kept share is reported), recent samples matched on window size, and label-free
   alert-gated self-calibration (the buffer admits only non-alerted full windows).

Both datasets are consumed; every number here is exploratory. Writes
output/calibration_policy_probes/report.json and refuses to overwrite it.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from run_entity_detector import write_json

from fedxmag.config import feature_groups_from_config, load_stage1_config
from fedxmag.m1.detection import auroc
from fedxmag.serialization import load_graph_samples

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/calibration_policy_probes"
DEVELOPMENT = ROOT / "data/processed/xiiotid_stage1_v3b_development"
AUDIT_BAND = ROOT / "data/processed/xiiotid_stage1_v3b_audit_band"
XIIOTID_PARTS = {
    "calibration": DEVELOPMENT / "calibration_normal.pt",
    "test_normal": DEVELOPMENT / "test_normal.pt",
    "test_attack": DEVELOPMENT / "test_attack.pt",
    "audit_normal": DEVELOPMENT / "audit_normal.pt",
    "audit_attack": AUDIT_BAND / "audit_attack.pt",
}
TON_SAMPLES = ROOT / "output/ton_iot_fpr_diagnosis/samples.csv"
TARGET = 0.05


def conformal(values, target: float = TARGET) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    rank = math.ceil((len(ordered) + 1) * (1 - target))
    return float(ordered[rank - 1]) if rank <= len(ordered) else math.nan


def oracle_tpr(negative: np.ndarray, positive: np.ndarray) -> float:
    return float(np.mean(positive > np.sort(negative)[math.ceil(0.95 * len(negative)) - 1]))


def xiiotid() -> dict[str, Any]:
    config = load_stage1_config(ROOT / "configs/stage1_m0_v3_development.yaml")
    groups = feature_groups_from_config(config)
    archived = torch.load(ROOT / "output/fusion_detector_v2/replay_scores.pt", weights_only=False)
    scores: dict[str, dict[str, np.ndarray]] = {"fusion": {}, "record_count": {}}
    categories: dict[str, list[list[str]]] = {}
    for name, path in XIIOTID_PARTS.items():
        samples = load_graph_samples(path, groups)
        scores["fusion"][name] = archived["fusion"][name].numpy()
        scores["record_count"][name] = np.log1p(np.asarray(
            [s.metadata["event_count"] - s.metadata["synthetic_reverse_event_count"]
             for s in samples], dtype=np.float64))
        categories[name] = [list(s.metadata.get("attack_categories", [])) for s in samples]
        if len(scores["fusion"][name]) != len(samples):
            raise ValueError(f"Archived scores do not align with {name}")
    report: dict[str, Any] = {}
    for split in ("test", "audit"):
        negative, positive = f"{split}_normal", f"{split}_attack"
        entry: dict[str, Any] = {"negatives": len(scores["fusion"][negative]),
                                 "positives": len(scores["fusion"][positive])}
        thresholds = {score: conformal(values["calibration"]) for score, values in scores.items()}
        for score, values in scores.items():
            entry[score] = {
                "auroc": auroc(values[negative], values[positive]),
                "tpr": float(np.mean(values[positive] > thresholds[score])),
                "fpr": float(np.mean(values[negative] > thresholds[score])),
                "oracle_tpr_at_eval_fpr_5pct": oracle_tpr(values[negative], values[positive]),
            }
        records = np.expm1(scores["record_count"][positive])
        entry["per_category"] = {}
        for kind in sorted({c for cs in categories[positive] for c in cs}):
            mask = np.asarray([kind in cs for cs in categories[positive]])
            entry["per_category"][kind] = {
                "windows": int(mask.sum()),
                "median_records": float(np.median(records[mask])),
                **{f"{score}_tpr": float(np.mean(values[positive][mask] > thresholds[score]))
                   for score, values in scores.items()},
            }
        normal_records = np.expm1(scores["record_count"][negative])
        entry["median_records_normal"] = float(np.median(normal_records))
        report[split] = entry
    return report


def ton_iot() -> dict[str, Any]:
    frame = pd.read_csv(TON_SAMPLES)
    frame["record_count"] = np.log1p(frame.records)
    frame["log_flows"] = np.log1p(frame.flows)
    calibration = frame[frame.role == "calibration"]
    frame["combined"] = np.maximum(
        frame.score / np.quantile(calibration.score, 0.95),
        frame.record_count / np.quantile(calibration.record_count, 0.95),
    )
    calibration = frame[frame.role == "calibration"]
    negative, positive = frame[frame.role == "negative"], frame[frame.role == "positive"]
    background = frame[frame.role != "positive"].sort_values(["capture", "minute"])
    scores = ("score", "record_count", "combined")

    def rates(flags: dict[str, list[bool]], captures: list[str],
              coverage: float = 1.0) -> dict[str, Any]:
        per = pd.Series(flags["negative"]).groupby(np.asarray(captures)).agg(["mean", "size"])
        return {"tpr": float(np.mean(flags["positive"])), "fpr": float(np.mean(flags["negative"])),
                "worst_capture_fpr": float(per["mean"].max()),
                "worst_capture_fpr_50_negatives": float(per["mean"][per["size"] >= 50].max()),
                "negative_coverage": coverage}

    def recent(score: str, memory: int, delay: int, nearest: int | None = None,
               size: str = "record_count", minimum: int | None = None) -> dict[str, Any]:
        """Thresholds from the last `memory` background samples ending `delay` minutes
        earlier (at least `minimum`, default `memory`); unsupported windows are left out."""

        flags: dict[str, list[bool]] = {"negative": [], "positive": []}
        captures: list[str] = []
        supported = 0
        for role, part in (("negative", negative), ("positive", positive)):
            for capture, rows in part.groupby("capture", sort=False):
                past = background[background.capture == capture]
                minutes = past.minute.to_numpy()
                values, sizes = past[score].to_numpy(), past[size].to_numpy()
                for minute, value, own in zip(rows.minute, rows[score], rows[size], strict=True):
                    keep = np.flatnonzero(minutes < minute - delay)[-memory:]
                    if len(keep) < (minimum or memory):
                        continue
                    if nearest is not None:
                        order = np.argsort(np.abs(sizes[keep] - own), kind="stable")
                        keep = keep[order[:nearest]]
                    flags[role].append(bool(value > conformal(values[keep])))
                    if role == "negative":
                        captures.append(capture)
                        supported += 1
        return rates(flags, captures, supported / len(negative))

    def gated(score: str, memory: int) -> dict[str, Any]:
        flags: dict[str, list[bool]] = {"negative": [], "positive": []}
        captures: list[str] = []
        for capture in sorted(frame.capture.unique()):
            buffer = calibration[calibration.capture == capture].sort_values("minute")[score]
            buffer = buffer.tolist()[-memory:]
            own_negative = negative[negative.capture == capture].set_index("minute")[score]
            own_positive = positive[positive.capture == capture].set_index("minute")[score]
            for minute in sorted(set(own_negative.index) | set(own_positive.index)):
                threshold = conformal(buffer)
                if minute in own_negative.index:
                    flags["negative"].append(bool(own_negative[minute] > threshold))
                    captures.append(capture)
                if minute in own_positive.index:
                    flags["positive"].append(bool(own_positive[minute] > threshold))
                stream = (own_positive[minute] if minute in own_positive.index
                          else own_negative[minute])
                if stream <= threshold:
                    buffer = [*buffer, stream][-memory:]
        return rates(flags, captures)

    report: dict[str, Any] = {"static_pooled": {}, "recent": {}, "recent_size_matched": {},
                              "alert_gated": {}}
    for score in scores:
        threshold = conformal(calibration[score])
        report["static_pooled"][score] = rates(
            {"negative": (negative[score] > threshold).tolist(),
             "positive": (positive[score] > threshold).tolist()},
            negative.capture.tolist())
        for memory in (20, 60, 100):
            for delay in (0, 30, 60, 120):
                report["recent"].setdefault(f"m{memory}_d{delay}", {})[score] = recent(
                    score, memory, delay)
        for delay in (0, 15, 30, 60, 120):
            report["recent"].setdefault(f"m20to60_d{delay}", {})[score] = recent(
                score, 60, delay, minimum=20)
        for memory in (20, 100):
            report["alert_gated"].setdefault(f"m{memory}", {})[score] = gated(score, memory)
    for size in ("record_count", "log_flows"):
        for memory, nearest in ((60, 20), (100, 40)):
            for delay in (0, 30, 120):
                report["recent_size_matched"][f"{size}_m{memory}_k{nearest}_d{delay}"] = recent(
                    "score", memory, delay, nearest, size)
    return report


def main() -> None:
    destination = OUTPUT / "report.json"
    if destination.exists():
        raise FileExistsError(destination)
    report = {
        "status": "post_hoc_exploratory_on_consumed_data",
        "xiiotid_detector_vs_record_count": xiiotid(),
        "ton_iot_threshold_policies": ton_iot(),
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json(destination, report)
    print(json.dumps(report, indent=1)[:6000])


if __name__ == "__main__":
    main()
