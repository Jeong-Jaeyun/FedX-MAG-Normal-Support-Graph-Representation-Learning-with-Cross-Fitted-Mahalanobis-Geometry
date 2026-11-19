"""Register fixed flow-channel candidates, then characterize them on consumed data.

``register`` binds the candidate file, code, archives, and inputs before any variant
score exists. ``evaluate`` runs once, refuses to run if a bound file changed, and
describes every candidate on the original split and the capture-disjoint folds.
Nothing is selected: all candidates carry forward to the new-data protocol.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from run_entity_detector import write_json
from run_fusion_detector import (
    BASE_CONFIG,
    RAW_CSV,
    ROLES,
    STUDY,
    load_windows,
    raw_graphs,
    verified_inputs,
)

from fedxmag.config import feature_groups_from_config, load_stage1_config
from fedxmag.confirmation import sha256_file
from fedxmag.flow_context import FLOW_VARIANTS
from fedxmag.fusion_detector import RawFlowDetector, TrafficWindows
from fedxmag.m1.detection import auroc, threshold_metrics
from fedxmag.preprocessing.scaling import FeatureStandardizer

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/flow_variant_candidates.yaml"
OUTPUT = ROOT / "output/flow_variants_20261027"
CODE = (
    "src/fedxmag/flow_context.py",
    "src/fedxmag/fusion_detector.py",
    "src/fedxmag/local_support.py",
    "src/fedxmag/m1/detection.py",
    "scripts/run_fusion_detector.py",
    "scripts/run_flow_variant_check.py",
)
ARCHIVES = (
    "generalization_scores.pt",
    "candidate_capture_scores.pt",
    "normal_commissioning_scores.pt",
)
MIN_CAPTURE_NORMALS = 10


def bindings() -> dict[str, Any]:
    return {
        "protocol_sha256": sha256_file(PROTOCOL),
        "code": {name: sha256_file(ROOT / name) for name in CODE},
        "archives": {name: sha256_file(STUDY / name) for name in ARCHIVES},
        "inputs": verified_inputs(),
    }


def register(output: Path) -> None:
    path = output / "registration.json"
    if path.exists() or (output / "report.json").exists():
        raise FileExistsError("The flow candidates are already registered")
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "status": "registered_before_any_flow_variant_score",
            "registered_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            **bindings(),
        },
    )
    print(json.dumps({"registered": str(path)}))


def capture_fpr(scores: np.ndarray, dates: np.ndarray, threshold: float) -> dict[str, Any]:
    return {
        str(date): {
            "normals": int(np.sum(dates == date)),
            "fpr": float(np.mean(scores[dates == date] > threshold)),
        }
        for date in sorted(set(dates.tolist()))
    }


def at_5pct(detector: RawFlowDetector, normal: np.ndarray, attack: np.ndarray) -> dict[str, Any]:
    threshold = detector.thresholds["0.05"]
    metrics = threshold_metrics(normal, attack, threshold)
    return {
        "auroc": auroc(normal, attack),
        "threshold": threshold,
        "tpr": metrics["tpr"],
        "fpr": metrics["fpr"],
        "tp": int(np.sum(attack > threshold)),
        "attacks": len(attack),
        "fp": int(np.sum(normal > threshold)),
        "normals": len(normal),
    }


def evaluate(output: Path) -> None:
    destination = output / "report.json"
    if destination.exists():
        raise FileExistsError(destination)
    registration = json.loads((output / "registration.json").read_text())
    if any(registration[key] != value for key, value in bindings().items()):
        raise ValueError("A registered candidate, code, archive, or input file changed")
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    variants = list(protocol["candidates"])
    if set(variants) != set(FLOW_VARIANTS):
        raise ValueError("Registered candidates differ from the implemented flow variants")
    groups = feature_groups_from_config(load_stage1_config(BASE_CONFIG))
    samples, flows_by_role, _ = load_windows(ROLES, groups, RAW_CSV)
    role_windows = {role: TrafficWindows(samples[role], flows_by_role[role]) for role in ROLES}
    graphs = [graph for role in ROLES for graph in samples[role]]
    raw = raw_graphs(graphs)
    flows = [matrix for role in ROLES for matrix in flows_by_role[role]]
    dates = np.asarray([graph.metadata["session_key"] for graph in graphs])
    archived = torch.load(STUDY / "generalization_scores.pt", weights_only=True)[
        "raw_node_plus_flow_knn"
    ]
    capture = torch.load(STUDY / "candidate_capture_scores.pt", weights_only=True)
    commissioning = torch.load(STUDY / "normal_commissioning_scores.pt", weights_only=True)
    report: dict[str, Any] = {
        "status": protocol["protocol"]["status"],
        "protocol_sha256": registration["protocol_sha256"],
        "selection": protocol["decision_rule"]["selection"],
        "variants": {},
    }
    control_checks: list[bool] = []
    saved: dict[str, Any] = {}
    for variant in variants:
        entry: dict[str, Any] = {"original_split": {}, "capture_folds": {"folds": {}}}
        saved[variant] = {}
        detector = RawFlowDetector.fit(
            groups,
            role_windows["train"],
            role_windows["selection"],
            role_windows["calibration"],
            flow_variant=variant,
        )
        threshold = detector.thresholds["0.05"]
        for band in ("test", "audit"):
            normal, channels, _ = detector.score(role_windows[f"{band}_normal"])
            attack = detector.score(role_windows[f"{band}_attack"])[0]
            saved[variant][f"{band}_normal"] = torch.from_numpy(normal)
            saved[variant][f"{band}_attack"] = torch.from_numpy(attack)
            raw_alarm = channels["raw_node"] / detector.scales["raw_node"] > threshold
            alarm = normal > threshold
            normal_dates = np.asarray(
                [sample.metadata["session_key"] for sample in samples[f"{band}_normal"]]
            )
            entry["original_split"][band] = {
                "auroc": auroc(normal, attack),
                "at_calibrated_fpr": {
                    target: None if value is None else threshold_metrics(normal, attack, value)
                    for target, value in detector.thresholds.items()
                },
                "false_positives_at_5pct": int(alarm.sum()),
                "flow_only_false_positives_at_5pct": int((alarm & ~raw_alarm).sum()),
                "per_capture_fpr_at_5pct": capture_fpr(normal, normal_dates, threshold),
            }
            if variant == "all":
                control_checks.extend(
                    np.array_equal(values, archived[f"{band}_{role}"].numpy())
                    for role, values in (("normal", normal), ("attack", attack))
                )
                covered = np.isin(normal_dates, detector.calibration_sessions)
                entry.setdefault("calibration_coverage", {})[band] = {
                    name: {
                        "normals": int(mask.sum()),
                        "fp": int(alarm[mask].sum()),
                        "fpr": float(alarm[mask].mean()) if mask.any() else None,
                        "captures": sorted({str(d) for d in normal_dates[mask]}),
                    }
                    for name, mask in (("in_calibration", covered), ("outside", ~covered))
                }
        pooled = {arm: dict.fromkeys(("tp", "attacks", "fp", "normals"), 0)
                  for arm in ("source_only", "commissioned")}
        for fold, stored in capture.items():
            indices = {role: values.tolist() for role, values in stored["indices"].items()}
            scaler = FeatureStandardizer.fit([raw[i] for i in indices["train"]])

            def windows(items: list[int], scaler=scaler) -> TrafficWindows:
                chosen = scaler.transform_many([raw[i] for i in items], groups)
                return TrafficWindows(chosen, [flows[i] for i in items])

            source = RawFlowDetector.fit(
                groups,
                windows(indices["train"]),
                windows(indices["selection"]),
                windows(indices["calibration"]),
                flow_variant=variant,
            )
            target = {r: v.tolist() for r, v in commissioning[fold]["target_indices"].items()}
            arms = {
                "source_only": source,
                "commissioned": source.commission(
                    windows(target["reference"]),
                    windows(target["calibration"]),
                    environment=f"capture_fold_{fold}",
                ),
            }
            evaluation = {role: windows(target[role]) for role in ("normal", "attack")}
            fold_entry = {}
            for arm, arm_detector in arms.items():
                normal = arm_detector.score(evaluation["normal"])[0]
                attack = arm_detector.score(evaluation["attack"])[0]
                saved[variant][f"fold_{fold}|{arm}|normal"] = torch.from_numpy(normal)
                saved[variant][f"fold_{fold}|{arm}|attack"] = torch.from_numpy(attack)
                summary = at_5pct(arm_detector, normal, attack)
                per_capture = capture_fpr(
                    normal, dates[target["normal"]], arm_detector.thresholds["0.05"]
                )
                eligible = [v["fpr"] for v in per_capture.values()
                            if v["normals"] >= MIN_CAPTURE_NORMALS]
                summary["max_capture_fpr"] = max(eligible) if eligible else None
                summary["per_capture_fpr"] = per_capture
                fold_entry[arm] = summary
                for key in pooled[arm]:
                    pooled[arm][key] += summary[key]
                if variant == "all":
                    name = "fixed_reference" if arm == "source_only" else "reference_added"
                    archive = commissioning[fold]["scores"][name]["fusion"]
                    control_checks.extend(
                        np.array_equal(values, archive[role].numpy())
                        for role, values in (("normal", normal), ("attack", attack))
                    )
            entry["capture_folds"]["folds"][fold] = fold_entry
        folds = entry["capture_folds"]["folds"]
        entry["capture_folds"]["pooled_at_5pct"] = {
            arm: {
                **counts,
                "tpr": counts["tp"] / counts["attacks"],
                "fpr": counts["fp"] / counts["normals"],
                "worst_fold_auroc": min(f[arm]["auroc"] for f in folds.values()),
                "max_fold_fpr": max(f[arm]["fpr"] for f in folds.values()),
                "max_capture_fpr": max(
                    (
                        f[arm]["max_capture_fpr"]
                        for f in folds.values()
                        if f[arm]["max_capture_fpr"] is not None
                    ),
                    default=None,
                ),
            }
            for arm, counts in pooled.items()
        }
        report["variants"][variant] = entry
    report["controls_reproduce_archive"] = bool(control_checks and all(control_checks))
    if not report["controls_reproduce_archive"]:
        raise RuntimeError("The control variant did not reproduce the archived scores")
    report["finished_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    with (output / "scores.pt").open("xb") as stream:
        torch.save(saved, stream)
    write_json(destination, report)
    for variant, entry in report["variants"].items():
        original = entry["original_split"]
        print(
            json.dumps(
                {
                    "variant": variant,
                    **{
                        band: {
                            "auroc": round(original[band]["auroc"], 3),
                            "tpr": round(original[band]["at_calibrated_fpr"]["0.05"]["tpr"], 3),
                            "fpr": round(original[band]["at_calibrated_fpr"]["0.05"]["fpr"], 3),
                            "flow_only_fp": original[band]["flow_only_false_positives_at_5pct"],
                        }
                        for band in ("test", "audit")
                    },
                    **{
                        arm: {
                            key: None if value is None else round(value, 3)
                            for key, value in summary.items()
                            if key in ("tpr", "fpr", "worst_fold_auroc", "max_capture_fpr")
                        }
                        for arm, summary in entry["capture_folds"]["pooled_at_5pct"].items()
                    },
                }
            ),
            flush=True,
        )
    print(json.dumps({"calibration_coverage": report["variants"]["all"]["calibration_coverage"]}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "evaluate"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase == "register":
        register(args.output_dir)
    else:
        evaluate(args.output_dir)


if __name__ == "__main__":
    main()
