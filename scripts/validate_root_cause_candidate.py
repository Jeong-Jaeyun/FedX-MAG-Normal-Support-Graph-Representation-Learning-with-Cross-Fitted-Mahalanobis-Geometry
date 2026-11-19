"""Validate the raw-node + flow candidate, including capture-disjoint failures.

Everything is exploratory on consumed data. No frozen source or prior result
is changed. The capture stress inverts the archived affine feature scaler and
refits it inside each training fold; the pretrained GNN is never used there.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import torch
from investigate_context import ROLES
from investigate_detection import DEFAULT_OUTPUT, PREPARED, ROOT, write_json
from summarize_root_cause import roc_at_fpr

from fedxmag.local_support import LocalSupport
from fedxmag.m1.detection import Standardizer, auroc, detection_summary
from fedxmag.m1.entity import aggregate, raw_entity_features
from fedxmag.preprocessing.scaling import FeatureStandardizer
from fedxmag.scoring import FrozenNodeDetector
from fedxmag.serialization import load_graph_samples


def interval(values):
    return {
        "mean": float(np.mean(values)),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
    }


def candidate_bootstrap(samples, output):
    context = torch.load(output / "context_scores.pt", weights_only=True)
    general = torch.load(output / "generalization_scores.pt", weights_only=True)
    groups = {
        "node_lw": [context[f"node_lw_seed_{s}"] for s in (17, 29, 41, 53, 67)],
        "flow_knn": [context["all_flow_knn"]],
        "raw_node_knn": [general["raw_node_knn"]],
        "raw_node_plus_flow_knn": [general["raw_node_plus_flow_knn"]],
    }
    report = {}
    for band in ("test", "audit"):
        dates = {
            role: [s.metadata["session_key"] for s in samples[f"{band}_{role}"]]
            for role in ("normal", "attack")
        }
        clusters = sorted(set(dates["normal"]) | set(dates["attack"]))
        indices = {
            role: np.asarray([clusters.index(d) for d in values]) for role, values in dates.items()
        }
        point, arrays = {}, {}
        for name, runs in groups.items():
            arrays[name] = [{role: run[f"{band}_{role}"].numpy() for role in dates} for run in runs]
            point[name] = {
                key: float(
                    np.mean([roc_at_fpr(r["normal"], r["attack"])[key] for r in arrays[name]])
                )
                for key in ("tpr_at_empirical_fpr_leq_5pct", "empirical_fpr")
            }
        rng = np.random.default_rng(20261026)
        draws = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), 2000)
        values = {name: {metric: [] for metric in ("auroc", "matched_tpr")} for name in groups}
        for draw in draws:
            rows = {
                role: np.repeat(np.arange(len(idx)), draw[idx]) for role, idx in indices.items()
            }
            if not all(len(r) for r in rows.values()):
                continue
            for name, runs in arrays.items():
                metrics = []
                for run in runs:
                    normal, attack = (run[role][rows[role]] for role in ("normal", "attack"))
                    metrics.append(
                        (
                            auroc(normal, attack),
                            roc_at_fpr(normal, attack)["tpr_at_empirical_fpr_leq_5pct"],
                        )
                    )
                for i, metric in enumerate(("auroc", "matched_tpr")):
                    values[name][metric].append(float(np.mean([m[i] for m in metrics])))
        differences = {}
        for other in ("node_lw", "flow_knn", "raw_node_knn"):
            differences[f"candidate_minus_{other}"] = {
                metric: interval(
                    np.asarray(values["raw_node_plus_flow_knn"][metric])
                    - np.asarray(values[other][metric])
                )
                for metric in ("auroc", "matched_tpr")
            }
        report[band] = {"points": point, "differences": differences, "date_clusters": len(clusters)}
        print(json.dumps({"band": band, **report[band]}), flush=True)
    return report


def capture_stress(samples, output, groups):
    archived = json.loads((output / "generalization_study.json").read_text())
    assignment = archived["capture_stress"]["assignment"]
    saved = torch.load(output / "flow_context_cache.pt", weights_only=True)["features"]
    graphs = [s for role in samples for s in samples[role]]
    flows = [t for role in samples for t in saved[role]]
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
    fold_ids = np.asarray([assignment[d] for d in dates])
    labels = np.asarray([s.label for s in graphs])
    result, score_archive = {}, {}
    for fold in range(5):
        eligible = np.flatnonzero((fold_ids != fold) & (fold_ids != (fold + 1) % 5) & (labels == 0))
        # Fixed hash split of eligible normal windows, independent of outcomes.
        order = sorted(
            eligible,
            key=lambda i: hashlib.sha256(
                f"{dates[i]}:{graphs[i].metadata['window_start_ns']}".encode()
            ).hexdigest(),
        )
        select_n = max(20, len(order) // 5)
        indices = {
            "selection": np.asarray(order[:select_n]),
            "train": np.asarray(order[select_n:]),
            "calibration": np.flatnonzero((fold_ids == (fold + 1) % 5) & (labels == 0)),
            "normal": np.flatnonzero((fold_ids == fold) & (labels == 0)),
            "attack": np.flatnonzero((fold_ids == fold) & (labels == 1)),
        }
        train_dates = set(dates[indices["train"]])
        cal_dates = set(dates[indices["calibration"]])
        eval_dates = set(dates[fold_ids == fold])
        assert train_dates.isdisjoint(cal_dates | eval_dates)
        assert cal_dates.isdisjoint(eval_dates)
        scaler = FeatureStandardizer.fit([raw[i] for i in indices["train"]])
        nodes = {
            role: raw_entity_features(scaler.transform_many([raw[i] for i in idx], groups)).node
            for role, idx in indices.items()
        }
        node_model = LocalSupport.fit(torch.cat(nodes["train"]), k=20)
        flow_ref = torch.cat([flows[i] for i in indices["train"]])
        flow_scaler = Standardizer.fit(flow_ref)
        flow_model = LocalSupport.fit(flow_scaler(flow_ref), k=20)
        scores = {"raw_node": {}, "flow": {}}
        for role, idx in indices.items():
            if role == "train":
                continue
            for name, model, rows in (
                ("raw_node", node_model, nodes[role]),
                ("flow", flow_model, [flow_scaler(flows[i]) for i in idx]),
            ):
                entity = model.score(torch.cat(rows))["knn"]
                scores[name][role] = aggregate(entity.split([len(t) for t in rows]), "top3_mean")
        scales = {
            name: max(float(np.quantile(score["selection"], 0.95)), 1e-10)
            for name, score in scores.items()
        }
        scores["fusion"] = {
            role: np.maximum(
                scores["raw_node"][role] / scales["raw_node"], scores["flow"][role] / scales["flow"]
            )
            for role in scores["raw_node"]
        }
        entry = {
            "counts": {role: len(idx) for role, idx in indices.items()},
            "metrics": {},
            "per_date": {},
        }
        for name, score in scores.items():
            metrics = detection_summary(
                score["calibration"], score["normal"], score["attack"], [0.05]
            )
            metrics.update(roc_at_fpr(score["normal"], score["attack"]))
            entry["metrics"][name] = metrics
            threshold = metrics["at_calibrated_fpr"]["0.05"]["threshold"]
            entry["per_date"][name] = {}
            for date in sorted(eval_dates):
                n = score["normal"][dates[indices["normal"]] == date]
                a = score["attack"][dates[indices["attack"]] == date]
                entry["per_date"][name][date] = {
                    "normal": len(n),
                    "attack": len(a),
                    "fpr": float(np.mean(n > threshold)) if len(n) else None,
                    "tpr": float(np.mean(a > threshold)) if len(a) else None,
                    "normal_median_score": float(np.median(n)) if len(n) else None,
                }
        result[str(fold)] = entry
        score_archive[str(fold)] = {
            "indices": {k: torch.from_numpy(v) for k, v in indices.items()},
            "scores": {
                k: {r: torch.from_numpy(v) for r, v in s.items()} for k, s in scores.items()
            },
        }
        print(json.dumps({"capture_fold": fold, "metrics": entry["metrics"]}), flush=True)
    with (output / "candidate_capture_scores.pt").open("xb") as stream:
        torch.save(score_archive, stream)
    return result


def main():
    output = DEFAULT_OUTPUT
    destination = output / "candidate_validation.json"
    if destination.exists() or (output / "candidate_capture_scores.pt").exists():
        raise FileExistsError("Candidate validation already exists")
    torch.set_num_threads(4)
    detector = FrozenNodeDetector.load(ROOT / "output/entity_detector/detector_seed_17.pt")
    samples = {role: load_graph_samples(path, detector.groups) for role, path in ROLES.items()}
    report = {
        "status": "exploratory_consumed_data_not_confirmatory",
        "matched_fpr_policy": "Evaluation-normal ROC diagnostic, not a deployment threshold",
        "bootstrap": candidate_bootstrap(samples, output),
        "capture_stress": capture_stress(samples, output, detector.groups),
    }
    write_json(destination, report)


if __name__ == "__main__":
    main()
