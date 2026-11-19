"""Fixed feature-group ablation of the failed capture-disjoint candidate.

Uses exactly the saved train/selection/calibration/evaluation indices. Every
feature group and fold is reported, with no attack-based tuning or selection.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from investigate_context import ROLES
from investigate_detection import DEFAULT_OUTPUT, write_json
from summarize_root_cause import roc_at_fpr

from fedxmag.local_support import LocalSupport
from fedxmag.m1.detection import Standardizer, detection_summary
from fedxmag.m1.entity import aggregate


def main():
    output = DEFAULT_OUTPUT
    path = output / "capture_feature_shift.json"
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(4)
    cache = torch.load(output / "flow_context_cache.pt", weights_only=True)
    flow = [t for role in ROLES for t in cache["features"][role]]
    archived = torch.load(output / "candidate_capture_scores.pt", weights_only=True)
    variants = {
        "without_protocol_service": [i for i in range(17) if not 1 <= i <= 6],
        "activity_flags": [0, *range(7, 13), 15, 16],
        "activity": [0, 12, 15, 16],
    }
    report = {"status": "exploratory_followup_on_capture_failure", "folds": {}}
    score_archive = {}
    for fold, saved in archived.items():
        idx = saved["indices"]
        raw = {role: score.numpy() for role, score in saved["scores"]["raw_node"].items()}
        raw_scale = max(float(np.quantile(raw["selection"], 0.95)), 1e-10)
        report["folds"][fold], score_archive[fold] = {}, {}
        for name, columns in variants.items():
            reference = torch.cat([flow[i][:, columns] for i in idx["train"]])
            scaler = Standardizer.fit(reference)
            model = LocalSupport.fit(scaler(reference), k=20)
            scores = {}
            for role, indices in idx.items():
                if role == "train":
                    continue
                rows = [flow[i][:, columns] for i in indices]
                entity = model.score(scaler(torch.cat(rows)))["knn"]
                scores[role] = aggregate(entity.split([len(t) for t in rows]), "top3_mean")
            scale = max(float(np.quantile(scores["selection"], 0.95)), 1e-10)
            fusion = {
                role: np.maximum(raw[role] / raw_scale, score / scale)
                for role, score in scores.items()
            }
            for method, arrays in ((name, scores), (f"raw_plus_{name}", fusion)):
                metrics = detection_summary(
                    arrays["calibration"], arrays["normal"], arrays["attack"], [0.05]
                )
                metrics.update(roc_at_fpr(arrays["normal"], arrays["attack"]))
                report["folds"][fold][method] = metrics
                score_archive[fold][method] = {r: torch.from_numpy(s) for r, s in arrays.items()}
        print(
            json.dumps(
                {
                    "fold": fold,
                    "methods": {
                        name: {"auroc": r["auroc"], **r["at_calibrated_fpr"]["0.05"]}
                        for name, r in report["folds"][fold].items()
                    },
                }
            ),
            flush=True,
        )
    with (output / "capture_feature_shift_scores.pt").open("xb") as stream:
        torch.save(score_archive, stream)
    write_json(path, report)


if __name__ == "__main__":
    main()
