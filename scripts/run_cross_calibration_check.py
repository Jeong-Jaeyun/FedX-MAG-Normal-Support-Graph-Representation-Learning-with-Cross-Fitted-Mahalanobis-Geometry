"""Pre-specified exploratory check of time-block cross-fitted commissioning calibration.

``register`` binds the protocol, code, archives, and inputs before any
cross-fitted fold score exists. ``evaluate`` runs once, refuses to run if a bound
file changed, and applies the protocol's fixed decision rule. Every window is
consumed X-IIoTID data, so the result is exploratory rather than confirmatory.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
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
from fedxmag.fusion_detector import RawFlowDetector, TrafficWindows
from fedxmag.m1.detection import auroc, threshold_metrics
from fedxmag.preprocessing.scaling import FeatureStandardizer

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/cross_calibration_protocol.yaml"
OUTPUT = ROOT / "output/cross_calibration_20261027"
CODE = (
    "src/fedxmag/fusion_detector.py",
    "src/fedxmag/flow_context.py",
    "src/fedxmag/local_support.py",
    "src/fedxmag/m1/detection.py",
    "scripts/run_fusion_detector.py",
    "scripts/run_cross_calibration_check.py",
)
ARCHIVES = (
    "candidate_capture_scores.pt",
    "normal_commissioning_scores.pt",
    "normal_commissioning.json",
)


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
        raise FileExistsError("This check is already registered")
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "status": "registered_before_any_cross_fitted_fold_score",
            "registered_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            **bindings(),
        },
    )
    print(json.dumps({"registered": str(path)}))


def binomial_upper_tail(successes: int, trials: int, rate: float) -> float:
    """P(X >= successes) for X ~ Binomial(trials, rate), summed in log space."""

    def term(i: int) -> float:
        return math.exp(
            math.lgamma(trials + 1) - math.lgamma(i + 1) - math.lgamma(trials - i + 1)
            + i * math.log(rate) + (trials - i) * math.log1p(-rate)
        )

    return float(min(1.0, sum(term(i) for i in range(successes, trials + 1))))


def arm_summary(
    detector: RawFlowDetector,
    normal: np.ndarray,
    attack: np.ndarray,
    normal_dates: np.ndarray,
    attack_dates: np.ndarray,
) -> dict[str, Any]:
    thresholds = detector.thresholds
    primary = thresholds["0.05"]
    return {
        "calibration_windows": len(detector.calibration_scores),
        "auroc": auroc(normal, attack),
        "thresholds": thresholds,
        "at_calibrated_fpr": {
            target: None if value is None else threshold_metrics(normal, attack, value)
            for target, value in thresholds.items()
        },
        "per_capture_at_5pct": {
            date: {
                "fp": int(np.sum(normal[normal_dates == date] > primary)),
                "normals": int(np.sum(normal_dates == date)),
                "tp": int(np.sum(attack[attack_dates == date] > primary)),
                "attacks": int(np.sum(attack_dates == date)),
            }
            for date in sorted(set(normal_dates) | set(attack_dates))
        },
    }


def evaluate(output: Path) -> None:
    destination = output / "report.json"
    if destination.exists():
        raise FileExistsError(destination)
    registration = json.loads((output / "registration.json").read_text())
    current = bindings()
    if any(registration[key] != value for key, value in current.items()):
        raise ValueError("A registered protocol, code, archive, or input file changed")
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    arms = {name: protocol["arms"][name].get("blocks") for name in ("primary", "sensitivity")}
    groups = feature_groups_from_config(load_stage1_config(BASE_CONFIG))
    samples, flows_by_role, _ = load_windows(ROLES, groups, RAW_CSV)
    graphs = [graph for role in ROLES for graph in samples[role]]
    raw = raw_graphs(graphs)
    flows = [matrix for role in ROLES for matrix in flows_by_role[role]]
    dates = np.asarray([graph.metadata["session_key"] for graph in graphs])
    capture = torch.load(STUDY / "candidate_capture_scores.pt", weights_only=True)
    commissioning = torch.load(STUDY / "normal_commissioning_scores.pt", weights_only=True)
    archived = json.loads((STUDY / "normal_commissioning.json").read_text())["folds"]
    report: dict[str, Any] = {
        "status": protocol["protocol"]["status"],
        "protocol_sha256": registration["protocol_sha256"],
        "folds": {},
    }
    saved_scores: dict[str, Any] = {}
    for fold, saved in capture.items():
        indices = {role: values.tolist() for role, values in saved["indices"].items()}
        scaler = FeatureStandardizer.fit([raw[i] for i in indices["train"]])

        def windows(items: list[int], scaler=scaler) -> TrafficWindows:
            chosen = scaler.transform_many([raw[i] for i in items], groups)
            return TrafficWindows(chosen, [flows[i] for i in items])

        source = RawFlowDetector.fit(
            groups,
            windows(indices["train"]),
            windows(indices["selection"]),
            windows(indices["calibration"]),
        )
        target = {r: v.tolist() for r, v in commissioning[fold]["target_indices"].items()}
        evaluation = {role: windows(target[role]) for role in ("normal", "attack")}
        detectors = {
            "baseline": source.commission(
                windows(target["reference"]),
                windows(target["calibration"]),
                environment=f"capture_fold_{fold}",
            )
        }
        pooled = windows(target["reference"] + target["calibration"])
        for name, blocks in arms.items():
            detectors[name] = source.commission(
                pooled, environment=f"capture_fold_{fold}", blocks=blocks
            )
        entry: dict[str, Any] = {"arms": {}}
        saved_scores[fold] = {}
        for name, detector in detectors.items():
            scores = {role: detector.score(items)[0] for role, items in evaluation.items()}
            saved_scores[fold][name] = {
                "calibration": torch.from_numpy(detector.calibration_scores.copy()),
                **{role: torch.from_numpy(values) for role, values in scores.items()},
            }
            entry["arms"][name] = arm_summary(
                detector,
                scores["normal"],
                scores["attack"],
                dates[target["normal"]],
                dates[target["attack"]],
            )
        # The baseline must equal the archived reference_added + target-calibration arm.
        archive = commissioning[fold]["scores"]["reference_added"]["fusion"]
        baseline = saved_scores[fold]["baseline"]
        entry["baseline_reproduces_archive"] = bool(
            all(torch.equal(baseline[role], archive[role]) for role in ("normal", "attack"))
            and torch.equal(baseline["calibration"], archive["calibration"])
            and entry["arms"]["baseline"]["at_calibrated_fpr"]["0.05"]
            == archived[fold]["metrics"]["reference_added__calibration"]["fusion"][
                "at_calibrated_fpr"
            ]["0.05"]
        )
        report["folds"][fold] = entry
    if not all(entry["baseline_reproduces_archive"] for entry in report["folds"].values()):
        raise RuntimeError("The baseline arm did not reproduce the archived commissioning study")

    # Pooled 5% metrics and the capture bootstrap of the TPR difference.
    captures: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for entry in report["folds"].values():
        for name, summary in entry["arms"].items():
            for date, counts in summary["per_capture_at_5pct"].items():
                captures[date][name] = counts
    names = list(captures)
    counts = {
        arm: {
            key: np.asarray([captures[d][arm][key] for d in names])
            for key in ("tp", "attacks", "fp", "normals")
        }
        for arm in ("baseline", *arms)
    }
    settings = protocol["metrics"]["bootstrap"]
    rng = np.random.default_rng(settings["seed"])
    draws = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), settings["resamples"])
    report["pooled_at_5pct"] = {}
    for arm in ("baseline", *arms):
        tp, attacks = counts[arm]["tp"].sum(), counts[arm]["attacks"].sum()
        fp, normals = counts[arm]["fp"].sum(), counts[arm]["normals"].sum()
        result: dict[str, Any] = {
            "tpr": float(tp / attacks),
            "fpr": float(fp / normals),
            "tp": int(tp),
            "attacks": int(attacks),
            "fp": int(fp),
            "normals": int(normals),
            "binomial_p_fpr_leq_5pct": binomial_upper_tail(int(fp), int(normals), 0.05),
            "max_fold_fpr": max(
                entry["arms"][arm]["at_calibrated_fpr"]["0.05"]["fpr"]
                for entry in report["folds"].values()
            ),
        }
        if arm != "baseline":
            differences = []
            for draw in draws:
                sampled = draw @ counts[arm]["attacks"]
                if sampled:
                    gained = draw @ counts[arm]["tp"] - draw @ counts["baseline"]["tp"]
                    differences.append(float(gained / sampled))
            result["tpr_difference_vs_baseline"] = {
                "point": result["tpr"] - float(counts["baseline"]["tp"].sum() / attacks),
                "lower": float(np.quantile(differences, 0.025)),
                "upper": float(np.quantile(differences, 0.975)),
                "resamples_used": len(differences),
            }
        report["pooled_at_5pct"][arm] = result

    primary = report["pooled_at_5pct"]["primary"]
    difference = primary["tpr_difference_vs_baseline"]
    safe = primary["binomial_p_fpr_leq_5pct"] >= 0.05 and primary["max_fold_fpr"] <= 0.10
    if safe and difference["point"] > 0 and difference["lower"] > 0:
        verdict = "supported"
    elif safe and difference["point"] > 0:
        verdict = "inconclusive"
    else:
        verdict = "not_supported"
    report["verdict"] = {
        "primary": verdict,
        "safety_conditions_hold": bool(safe),
        "rule": protocol["decision_rule"],
    }
    report["finished_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    with (output / "scores.pt").open("xb") as stream:
        torch.save(saved_scores, stream)
    write_json(destination, report)
    for fold, entry in report["folds"].items():
        print(
            json.dumps(
                {
                    "fold": fold,
                    **{
                        name: {
                            "n_cal": summary["calibration_windows"],
                            "auroc": round(summary["auroc"], 3),
                            **{
                                m: round(summary["at_calibrated_fpr"]["0.05"][m], 3)
                                for m in ("tpr", "fpr")
                            },
                        }
                        for name, summary in entry["arms"].items()
                    },
                }
            ),
            flush=True,
        )
    print(json.dumps({"pooled": report["pooled_at_5pct"], "verdict": report["verdict"]["primary"]}))


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
