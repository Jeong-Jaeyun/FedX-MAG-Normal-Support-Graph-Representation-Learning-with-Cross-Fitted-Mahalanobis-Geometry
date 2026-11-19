"""Counterfactual clean commissioning: isolate reference coverage and calibration.

This is a conditional remedy, not a zero-shot detector or a clean-data discovery
algorithm. Ground-truth normal windows simulate independently vetted commissioning
traffic. Attack-only/low-support captures are explicitly unsupported. No attack
before the end of commissioning is included in the evaluated future period.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import torch
from investigate_context import ROLES
from investigate_detection import DEFAULT_OUTPUT, PREPARED, ROOT, write_json

from fedxmag.local_support import LocalSupport
from fedxmag.m1.detection import Standardizer, detection_summary
from fedxmag.m1.entity import aggregate, raw_entity_features
from fedxmag.preprocessing.scaling import FeatureStandardizer
from fedxmag.scoring import FrozenNodeDetector
from fedxmag.serialization import load_graph_samples


def main():
    output = DEFAULT_OUTPUT
    path = output / "normal_commissioning.json"
    if path.exists():
        raise FileExistsError(path)
    torch.set_num_threads(4)
    detector = FrozenNodeDetector.load(ROOT / "output/entity_detector/detector_seed_17.pt")
    samples = {role: load_graph_samples(p, detector.groups) for role, p in ROLES.items()}
    graphs = [s for role in ROLES for s in samples[role]]
    cache = torch.load(output / "flow_context_cache.pt", weights_only=True)["features"]
    flows = [t for role in ROLES for t in cache[role]]
    archived = torch.load(output / "candidate_capture_scores.pt", weights_only=True)
    old = FeatureStandardizer.from_state_dict(
        torch.load(PREPARED / "standardizer.pt", weights_only=True)
    )
    raw = []
    for graph in graphs:
        fields = {}
        for kind in ("node", "edge"):
            values = getattr(graph, f"{kind}_features").double()
            values = (
                values * getattr(old, f"{kind}_scale").double()
                + getattr(old, f"{kind}_mean").double()
            )
            fields[f"{kind}_features"] = values.masked_fill(~getattr(graph, f"{kind}_available"), 0)
        raw.append(replace(graph, **fields))
    dates = np.asarray([s.metadata["session_key"] for s in graphs])
    times = np.asarray([s.metadata["window_start_ns"] for s in graphs])
    report = {
        "status": "exploratory_conditional_on_vetted_target_normal_commissioning",
        "rule": (
            "At least 10 normal windows/capture; earliest 20% reference, next 20% calibration; "
            "later normal and attack evaluation only. "
            "Minimum 19 pooled target calibration windows/fold."
        ),
        "folds": {},
    }
    score_archive = {}
    for fold, saved in archived.items():
        idx = {role: indices.numpy() for role, indices in saved["indices"].items()}
        target = {role: [] for role in ("reference", "calibration", "normal", "attack")}
        coverage = {}
        for date in sorted(set(dates[idx["normal"]]) | set(dates[idx["attack"]])):
            normal = sorted((i for i in idx["normal"] if dates[i] == date), key=lambda i: times[i])
            attack = [i for i in idx["attack"] if dates[i] == date]
            if len(normal) < 10:
                coverage[date] = {
                    "status": "unsupported_normal_coverage",
                    "normal": len(normal),
                    "attack": len(attack),
                }
                continue
            count = len(normal) // 5
            ref, cal, later = normal[:count], normal[count : 2 * count], normal[2 * count :]
            cutoff = times[cal[-1]]
            evaluated = [i for i in attack if times[i] > cutoff]
            assert max(times[ref]) < min(times[cal])
            assert min(times[later]) > cutoff
            for role, values in (
                ("reference", ref),
                ("calibration", cal),
                ("normal", later),
                ("attack", evaluated),
            ):
                target[role].extend(values)
            coverage[date] = {
                "status": "supported",
                "reference": len(ref),
                "calibration": len(cal),
                "normal": len(later),
                "attack": len(evaluated),
                "earlier_attack_excluded": len(attack) - len(evaluated),
                "cutoff_ns": int(cutoff),
            }
        entry = {"coverage": coverage, "counts": {r: len(v) for r, v in target.items()}}
        if len(target["calibration"]) < 19 or not target["attack"]:
            entry["status"] = "insufficient_pooled_target_calibration_or_attacks"
            report["folds"][fold] = entry
            continue
        scaler = FeatureStandardizer.fit([raw[i] for i in idx["train"]])
        flow_scaler = Standardizer.fit(torch.cat([flows[i] for i in idx["train"]]))
        roles = {
            **{
                f"source_{r}": list(v)
                for r, v in idx.items()
                if r in ("train", "selection", "calibration")
            },
            **target,
        }
        nodes = {
            role: raw_entity_features(
                scaler.transform_many([raw[i] for i in indices], detector.groups)
            ).node
            for role, indices in roles.items()
        }
        flow_rows = {
            role: [flow_scaler(flows[i]) for i in indices] for role, indices in roles.items()
        }
        # Scalers and fusion weights are held fixed while the reference grows.
        scales = {
            name: float(np.quantile(saved["scores"][name]["selection"].numpy(), 0.95))
            for name in ("raw_node", "flow")
        }
        all_scores = {}
        for adapt in (False, True):
            scores = {"raw_node": {}, "flow": {}}
            for name, rows in (("raw_node", nodes), ("flow", flow_rows)):
                reference = rows["source_train"] + (rows["reference"] if adapt else [])
                model = LocalSupport.fit(torch.cat(reference), k=20)
                for role in ("source_calibration", "calibration", "normal", "attack"):
                    entity = model.score(torch.cat(rows[role]))["knn"]
                    scores[name][role] = aggregate(
                        entity.split([len(t) for t in rows[role]]), "top3_mean"
                    )
            scores["fusion"] = {
                role: np.maximum(
                    scores["raw_node"][role] / scales["raw_node"],
                    scores["flow"][role] / scales["flow"],
                )
                for role in scores["raw_node"]
            }
            all_scores["reference_added" if adapt else "fixed_reference"] = scores
        entry["status"] = "evaluated"
        entry["metrics"] = {}
        for reference_name, scores in all_scores.items():
            for calibration in ("source_calibration", "calibration"):
                key = f"{reference_name}__{calibration}"
                entry["metrics"][key] = {
                    name: detection_summary(
                        values[calibration], values["normal"], values["attack"], [0.05]
                    )
                    for name, values in scores.items()
                }
        report["folds"][fold] = entry
        score_archive[fold] = {
            "target_indices": {r: torch.tensor(v) for r, v in target.items()},
            "scores": {
                a: {n: {r: torch.from_numpy(v) for r, v in s.items()} for n, s in methods.items()}
                for a, methods in all_scores.items()
            },
        }
        print(
            json.dumps(
                {
                    "fold": fold,
                    "counts": entry["counts"],
                    "fusion": {name: values["fusion"] for name, values in entry["metrics"].items()},
                }
            ),
            flush=True,
        )
    write_json(path, report)
    with (output / "normal_commissioning_scores.pt").open("xb") as stream:
        torch.save(score_archive, stream)


if __name__ == "__main__":
    main()
