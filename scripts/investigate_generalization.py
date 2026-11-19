"""Exploratory raw-feature fusion and disjoint-capture stress checks.

The capture stress uses only archived windows: normal windows from all original
roles and attacks from the consumed test/audit bands. Thus capture-disjoint
fitting is real, but this is neither a new untouched dataset nor a complete
capture reconstruction. No pretrained GNN is used in this stress check because
its original training included those capture dates.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from investigate_context import ROLES
from investigate_detection import DEFAULT_OUTPUT, ROOT, write_json

from fedxmag.local_support import LocalSupport
from fedxmag.m1.detection import Standardizer, detection_summary
from fedxmag.m1.entity import aggregate, raw_entity_features, stratified_aurocs
from fedxmag.scoring import FPR_TARGETS, FrozenNodeDetector
from fedxmag.serialization import load_graph_samples


def main() -> None:
    output = DEFAULT_OUTPUT
    destination = output / "generalization_study.json"
    if destination.exists():
        raise FileExistsError(destination)
    torch.set_num_threads(4)
    detector = FrozenNodeDetector.load(ROOT / "output/entity_detector/detector_seed_17.pt")
    samples = {role: load_graph_samples(path, detector.groups) for role, path in ROLES.items()}
    flow = torch.load(output / "flow_context_cache.pt", weights_only=True)["features"]
    context_scores = torch.load(output / "context_scores.pt", weights_only=True)
    information_scores = torch.load(output / "information_loss_scores.pt", weights_only=True)
    raw_model = LocalSupport.fit(raw_entity_features(samples["train"]).stacked("node"), k=20)
    raw_selection = raw_entity_features(samples["selection"]).node
    select_scores = raw_model.score(torch.cat(raw_selection))["knn"]
    raw = {role: value.numpy() for role, value in information_scores["v3b_raw_node_knn"].items()}
    raw["selection"] = aggregate(select_scores.split([len(t) for t in raw_selection]), "top3_mean")
    raw_scale = float(np.quantile(raw["selection"], 0.95))
    context = {role: value.numpy() for role, value in context_scores["all_flow_knn"].items()}
    flow_scale = float(np.quantile(context["selection"], 0.95))
    fused = {role: np.maximum(raw[role] / raw_scale, context[role] / flow_scale) for role in raw}
    score_methods = {"raw_node_knn": raw, "raw_node_plus_flow_knn": fused}
    for seed in (17, 29, 41, 53, 67):
        name = f"node_plus_all_flow_knn_seed_{seed}"
        score_methods[f"raw_plus_learned_plus_flow_seed_{seed}"] = {
            role: np.maximum(fused[role], context_scores[name][role].numpy()) for role in fused
        }
    report = {
        "status": "exploratory_not_independent_confirmation",
        "fusions": {},
        "capture_stress": {"folds": {}, "assignment": {}},
    }
    for band in ("test", "audit"):
        info_path = ROOT / (
            "output/diagnosis/test_band_windows.json"
            if band == "test"
            else "data/processed/xiiotid_stage1_v3b_audit_band/audit_attack_windows.json"
        )
        lookup = {
            (w["session"], w["window_start_ns"]): w for w in json.loads(info_path.read_text())
        }
        normal, attack = samples[f"{band}_normal"], samples[f"{band}_attack"]
        ctx = {
            "normal_nodes": np.asarray([s.node_features.shape[0] for s in normal]),
            "attack_nodes": np.asarray([s.node_features.shape[0] for s in attack]),
            "normal_dates": np.asarray([s.metadata["session_key"] for s in normal]),
            "attack_dates": np.asarray([s.metadata["session_key"] for s in attack]),
            "attack_fraction": np.asarray(
                [
                    lookup[(s.metadata["session_key"], s.metadata["window_start_ns"])][
                        "attack_fraction"
                    ]
                    for s in attack
                ]
            ),
        }
        for name, scores in score_methods.items():
            result = detection_summary(
                scores["calibration"],
                scores[f"{band}_normal"],
                scores[f"{band}_attack"],
                FPR_TARGETS,
            )
            result.update(
                stratified_aurocs(scores[f"{band}_normal"], scores[f"{band}_attack"], **ctx)
            )
            report["fusions"].setdefault(name, {})[band] = result
            print(
                json.dumps(
                    {
                        "method": name,
                        "band": band,
                        "auroc": result["auroc"],
                        "within_date": result["within_date_auroc"],
                        "rates": result["at_calibrated_fpr"]["0.05"],
                    }
                ),
                flush=True,
            )

    # Hash ordering and round-robin folds depend only on capture IDs, not labels.
    all_samples = [s for graphs in samples.values() for s in graphs]
    all_flow = [row for rows in flow.values() for row in rows]
    # Align explicitly: torch cache dict ordering is not a data alignment contract.
    all_flow = [row for role in samples for row in flow[role]]
    keys = [(s.metadata["session_key"], s.metadata["window_start_ns"]) for s in all_samples]
    if len(keys) != len(set(keys)):
        raise ValueError("Archive roles contain duplicate windows")
    captures = sorted(
        {s.metadata["session_key"] for s in all_samples},
        key=lambda name: hashlib.sha256(name.encode()).hexdigest(),
    )
    assignment = {name: index % 5 for index, name in enumerate(captures)}
    report["capture_stress"]["assignment"] = assignment
    fold_ids = np.asarray([assignment[s.metadata["session_key"]] for s in all_samples])
    labels = np.asarray([s.label for s in all_samples])
    for fold in range(5):
        cal_fold = (fold + 1) % 5
        train_indices = np.flatnonzero((fold_ids != fold) & (fold_ids != cal_fold) & (labels == 0))
        cal_indices = np.flatnonzero((fold_ids == cal_fold) & (labels == 0))
        normal_indices = np.flatnonzero((fold_ids == fold) & (labels == 0))
        attack_indices = np.flatnonzero((fold_ids == fold) & (labels == 1))
        entry = {
            "counts": {
                "train_normal": len(train_indices),
                "calibration_normal": len(cal_indices),
                "eval_normal": len(normal_indices),
                "eval_attack": len(attack_indices),
            },
            "train_captures": sorted(
                {all_samples[i].metadata["session_key"] for i in train_indices}
            ),
            "calibration_captures": sorted(
                {all_samples[i].metadata["session_key"] for i in cal_indices}
            ),
            "evaluation_captures": sorted(
                {all_samples[i].metadata["session_key"] for i in np.flatnonzero(fold_ids == fold)}
            ),
        }
        assert set(entry["train_captures"]).isdisjoint(entry["evaluation_captures"])
        assert set(entry["calibration_captures"]).isdisjoint(entry["evaluation_captures"])
        if len(cal_indices) < 19 or not len(normal_indices) or not len(attack_indices):
            entry["status"] = "insufficient_normal_calibration_or_evaluation_support"
        else:
            reference = torch.cat([all_flow[i] for i in train_indices])
            scaler = Standardizer.fit(reference)
            model = LocalSupport.fit(scaler(reference), k=20)
            scores = {}
            for role, indices in (
                ("calibration", cal_indices),
                ("normal", normal_indices),
                ("attack", attack_indices),
            ):
                rows = [all_flow[i] for i in indices]
                entity = model.score(scaler(torch.cat(rows)))["knn"]
                scores[role] = aggregate(entity.split([len(t) for t in rows]), "top3_mean")
            entry["status"] = "evaluated"
            entry["metrics"] = detection_summary(
                scores["calibration"], scores["normal"], scores["attack"], [0.05]
            )
        report["capture_stress"]["folds"][str(fold)] = entry
        print(
            json.dumps(
                {
                    "capture_fold": fold,
                    "status": entry["status"],
                    "counts": entry["counts"],
                    "metrics": entry.get("metrics"),
                }
            ),
            flush=True,
        )
    write_json(destination, report)
    with (output / "generalization_scores.pt").open("xb") as stream:
        torch.save(
            {
                name: {role: torch.from_numpy(v) for role, v in scores.items()}
                for name, scores in score_methods.items()
            },
            stream,
        )


if __name__ == "__main__":
    main()
