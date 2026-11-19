"""Compare mechanisms at matched FPR and bootstrap paired capture dates.

Matched-FPR numbers use evaluation-normal scores and are diagnostic ROC
statistics only. They are never used as deployed thresholds. Operational
thresholds remain fixed by calibration-normal, including in the paired
bootstrap, which resamples that calibration role as well.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from investigate_detection import DEFAULT_OUTPUT, ROLE_PATHS, ROOT, write_json

from fedxmag.m1.detection import (
    cluster_bootstrap_detection,
    detection_summary,
)
from fedxmag.scoring import FPR_TARGETS, FrozenNodeDetector
from fedxmag.serialization import load_graph_samples


def roc_at_fpr(normal: np.ndarray, attack: np.ndarray, target: float = 0.05) -> dict:
    threshold = float(np.quantile(normal, 1 - target, method="higher"))
    return {
        "tpr_at_empirical_fpr_leq_5pct": float(np.mean(attack > threshold)),
        "empirical_fpr": float(np.mean(normal > threshold)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    output = args.output_dir
    if (output / "root_cause_summary.json").exists():
        raise FileExistsError("Root-cause summary already exists")
    torch.set_num_threads(4)
    context_scores = torch.load(output / "context_scores.pt", weights_only=True)
    context_report = json.loads((output / "context_study.json").read_text())
    mechanisms = {
        seed: torch.load(output / f"mechanism_scores_seed_{seed}.pt", weights_only=True)
        for seed in (17, 29, 41, 53, 67)
    }
    reports = {
        seed: json.loads((output / f"mechanisms_seed_{seed}.json").read_text())
        for seed in mechanisms
    }
    methods = {}
    for name in (
        "mahalanobis|top3_mean",
        "mahalanobis|max",
        "knn20_euclidean|top3_mean",
        "knn20_whitened|top3_mean",
        "lof20_euclidean|top3_mean",
    ):
        methods[name] = {str(seed): run["graph"][name] for seed, run in mechanisms.items()}
    for name in ("all_flow_lw", "all_flow_knn", "without_protocol_service_knn", "flags_only_knn"):
        methods[name] = {"fixed": context_scores[name]}
    for name in ("node_plus_all_flow_lw", "node_plus_all_flow_knn"):
        methods[name] = {str(seed): context_scores[f"{name}_seed_{seed}"] for seed in mechanisms}
    detector = FrozenNodeDetector.load(ROOT / "output/entity_detector/detector_seed_17.pt")
    samples = {role: load_graph_samples(path, detector.groups) for role, path in ROLE_PATHS.items()}
    summary = {
        "status": "exploratory_root_cause_study",
        "datasets": {},
        "misses": {},
        "metric_policy": "Mean seed metrics, no seed ensemble; empirical FPR is diagnostic",
    }
    pairs = [
        ("node_plus_all_flow_knn", "mahalanobis|top3_mean"),
        ("all_flow_knn", "all_flow_lw"),
        ("node_plus_all_flow_knn", "all_flow_knn"),
        ("knn20_whitened|top3_mean", "mahalanobis|top3_mean"),
    ]
    for band in ("test", "audit"):
        table, score_map, members = {}, {}, {}
        for name, runs in methods.items():
            per_seed = []
            members[name] = []
            for seed, scores in runs.items():
                arrays = {
                    role: scores[role if role == "calibration" else f"{band}_{role}"].numpy()
                    for role in ("calibration", "normal", "attack")
                }
                key = f"{name}#{seed}"
                score_map[key] = arrays
                members[name].append(key)
                report = detection_summary(
                    arrays["calibration"], arrays["normal"], arrays["attack"], FPR_TARGETS
                )
                item = {"auroc": report["auroc"], **roc_at_fpr(arrays["normal"], arrays["attack"])}
                for target in FPR_TARGETS:
                    for metric in ("tpr", "fpr"):
                        item[f"{metric}@{target:g}"] = report["at_calibrated_fpr"][f"{target:g}"][
                            metric
                        ]
                per_seed.append(item)
            table[name] = {
                metric: float(np.mean([r[metric] for r in per_seed])) for metric in per_seed[0]
            }
            if name in context_report["methods"]:
                table[name]["within_date_auroc"] = context_report["methods"][name][band][
                    "within_date_auroc"
                ]
            elif name.startswith("node_plus_"):
                table[name]["within_date_auroc"] = float(
                    np.mean(
                        [
                            context_report["methods"][f"{name}_seed_{seed}"][band][
                                "within_date_auroc"
                            ]
                            for seed in mechanisms
                        ]
                    )
                )
            else:
                table[name]["within_date_auroc"] = float(
                    np.mean(
                        [
                            reports[seed]["methods"][name][band]["within_date_auroc"]
                            for seed in mechanisms
                        ]
                    )
                )
        clusters = {
            role: [
                s.metadata["session_key"]
                for s in samples[role if role == "calibration" else f"{band}_{role}"]
            ]
            for role in ("calibration", "normal", "attack")
        }
        # Pair comparisons use the whole predeclared ladder; every outcome is kept.
        print(json.dumps({"phase": "bootstrap", "band": band}), flush=True)
        bootstrap = cluster_bootstrap_detection(
            score_map, members, clusters, pairs, [0.05], args.bootstrap, 20261026
        )
        summary["datasets"][band] = {"table": table, "bootstrap": bootstrap}
        summary["misses"][band] = {
            "mean_missed_attack_top1_fraction": float(
                np.mean(
                    [
                        reports[seed]["methods"]["mahalanobis|top3_mean"][band]["miss_analysis"][
                            "missed_with_attack_top1_fraction"
                        ]
                        for seed in mechanisms
                    ]
                )
            ),
        }
        print(
            json.dumps({"band": band, "table": table, "paired": bootstrap["differences"]}),
            flush=True,
        )
    write_json(output / "root_cause_summary.json", summary)


if __name__ == "__main__":
    main()
