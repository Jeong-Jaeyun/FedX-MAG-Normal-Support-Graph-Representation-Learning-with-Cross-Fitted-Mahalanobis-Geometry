"""Paired date bootstrap of ROC sensitivity at the same empirical FPR.

This diagnostic intentionally uses evaluation-normal thresholds. It measures
ranking quality at equal false-positive cost and does not calibrate deployment.
"""

from __future__ import annotations

import numpy as np
import torch
from investigate_detection import DEFAULT_OUTPUT, ROLE_PATHS, ROOT, write_json
from summarize_root_cause import roc_at_fpr

from fedxmag.scoring import FrozenNodeDetector
from fedxmag.serialization import load_graph_samples


def main() -> None:
    output = DEFAULT_OUTPUT
    destination = output / "matched_fpr_bootstrap.json"
    if destination.exists():
        raise FileExistsError(destination)
    context = torch.load(output / "context_scores.pt", weights_only=True)
    seeds = (17, 29, 41, 53, 67)
    detector = FrozenNodeDetector.load(ROOT / "output/entity_detector/detector_seed_17.pt")
    report = {"status": "diagnostic_matched_FPR_not_deployment_thresholds", "bands": {}}
    for band in ("test", "audit"):
        samples = {
            role: load_graph_samples(ROLE_PATHS[f"{band}_{role}"], detector.groups)
            for role in ("normal", "attack")
        }
        dates = {
            role: [s.metadata["session_key"] for s in graphs] for role, graphs in samples.items()
        }
        clusters = sorted(set(dates["normal"]) | set(dates["attack"]))
        indices = {
            role: np.asarray([clusters.index(d) for d in values]) for role, values in dates.items()
        }
        groups = {
            "node": [f"node_lw_seed_{s}" for s in seeds],
            "flow": ["all_flow_knn"],
            "fusion": [f"node_plus_all_flow_knn_seed_{s}" for s in seeds],
        }
        rng = np.random.default_rng(20261026)
        draws = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=2000)
        values = {name: [] for name in groups}
        for draw in draws:
            rows = {
                role: np.repeat(np.arange(len(idx)), draw[idx]) for role, idx in indices.items()
            }
            if not all(len(r) for r in rows.values()):
                continue
            for name, members in groups.items():
                score = []
                for member in members:
                    n = context[member][f"{band}_normal"].numpy()[rows["normal"]]
                    a = context[member][f"{band}_attack"].numpy()[rows["attack"]]
                    score.append(roc_at_fpr(n, a)["tpr_at_empirical_fpr_leq_5pct"])
                values[name].append(float(np.mean(score)))
        differences = {}
        for left, right in (("fusion", "node"), ("flow", "node"), ("fusion", "flow")):
            delta = np.asarray(values[left]) - np.asarray(values[right])
            differences[f"{left}_minus_{right}"] = {
                "mean": float(delta.mean()),
                "lower": float(np.quantile(delta, 0.025)),
                "upper": float(np.quantile(delta, 0.975)),
            }
        report["bands"][band] = {"resamples": len(values["node"]), "differences": differences}
    write_json(destination, report)
    print(report)


if __name__ == "__main__":
    main()
