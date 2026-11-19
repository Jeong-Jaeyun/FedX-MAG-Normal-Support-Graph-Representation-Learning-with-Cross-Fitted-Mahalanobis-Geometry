"""Prospectively ordered daily normal-support adaptation for actual connections.

For each target date, fit only on source-train normals and verified normals
completed before the current date's reference cutoff. Calibration is the next
time block. Feedback from earlier evaluation dates may enter later fits because
it is available then; no fit can change an earlier prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .contrastive_energy import DependencyEnergy, train
from .experiments import knn_scores, sha256, write_json
from .geometry import ConditionalMahalanobis
from .record_calibration import cluster_difference, replay
from .record_experiment import corruption_groups


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_path = Path(args.data).resolve()
    with np.load(data_path) as payload:
        data = {key: payload[key] for key in payload.files}
    dataset = str(data.get("dataset", "ton"))
    offset = int(data.get("utc_offset_seconds", 10 * 3600))
    if dataset == "xiiotid":
        days = np.unique((data["timestamp"] + offset) // 86400)
        cutoff = int(days[int(0.6 * len(days))]) * 86400 - offset
        end = (int(days[-1]) + 1) * 86400 - offset
        dates = [str(np.datetime64(second + offset, "s")) for second in (cutoff, end)]
    else:
        dates = ("2019-04-23", "2019-05-01") if dataset == "ton" else ("2017-07-06", "2017-07-08")
        cutoff, end = [int(np.datetime64(day, "s").astype(np.int64)) - offset for day in dates]
    timestamps, labels, captures = data["observed_at"], data["label"], data["capture"]
    source = np.flatnonzero((timestamps < cutoff) & (labels == 0))
    source = source[np.argsort(timestamps[source], kind="stable")]
    a, b = int(0.7 * len(source)), int(0.85 * len(source))
    source_train, selection = source[:a], source[a:b]
    target = np.flatnonzero((timestamps >= cutoff) & (timestamps < end))
    day_roles, all_evaluation = {}, []
    for capture in sorted(set(captures[target])):
        members = target[captures[target] == capture]
        members = members[np.argsort(timestamps[members], kind="stable")]
        first, second = np.quantile(timestamps[members], [0.2, 0.4], method="lower")
        calibration = members[
            (timestamps[members] > first) & (timestamps[members] <= second) & (labels[members] == 0)
        ]
        evaluation = members[timestamps[members] > second]
        # All target normals completed before this day's reference cutoff are
        # already verified, including earlier dates' feedback.
        reference = target[(timestamps[target] <= first) & (labels[target] == 0)]
        day_roles[int(capture)] = {
            "reference": reference,
            "calibration": calibration,
            "evaluation": evaluation,
            "reference_cutoff": int(first),
        }
        all_evaluation.extend(evaluation.tolist())
    all_evaluation = np.asarray(all_evaluation, dtype=int)
    if len(all_evaluation) > args.max_evaluation:
        all_evaluation = all_evaluation[
            np.argsort(data["signature"][all_evaluation])[: args.max_evaluation]
        ]
    selected = set(all_evaluation.tolist())
    for role in day_roles.values():
        role["evaluation"] = np.asarray([i for i in role["evaluation"] if i in selected], dtype=int)
    write_json(
        output / "registration.json",
        {
            "status": "chronologically_ordered_connection_development",
            "args": vars(args),
            "data_sha256": sha256(data_path),
            "dataset": dataset,
            "invariant": (
                "max(model reference observed_at) < min(current-day evaluation observed_at)"
            ),
            "feedback": "verified normals; score all same-second records before admitting feedback",
            "fit_reference_cap": args.max_reference,
            "evaluation_records": len(all_evaluation),
            "target_dates": dates,
            "source_selection_is_never_fitted": True,
        },
    )
    features = torch.from_numpy(data["features"])
    all_scores, all_flags, all_valid, completed = {}, {}, {}, {}
    position = {index: item for item, index in enumerate(all_evaluation)}
    for capture, role in day_roles.items():
        indices = role["evaluation"]
        if not len(indices):
            continue
        reference = np.concatenate((source_train, role["reference"]))
        # Preserve the exact temporal audit before any labels from the suffix can
        # be supplied to training or source-normal model selection.
        if int(timestamps[reference].max()) >= int(timestamps[indices].min()):
            raise ValueError("Future reference data entered a chronological daily fit")
        if np.intersect1d(reference, selection).size:
            raise ValueError("Selection-normal observations entered model fitting")
        reference_hash = data["signature"][reference]
        if len(reference) > args.max_reference:
            reference = reference[np.argsort(reference_hash)[: args.max_reference]]
        matrices_indices = {
            "train": reference,
            "selection": selection,
            "calibration": role["calibration"],
            "evaluation": indices,
        }
        normal = features[reference]
        mean, scale = normal.mean(dim=0), normal.std(dim=0, unbiased=False).clamp_min(0.05)
        matrices = {
            name: (features[members] - mean) / scale for name, members in matrices_indices.items()
        }
        print(
            json.dumps(
                {
                    "phase": "daily_fit",
                    "capture": capture,
                    "normal_reference": len(reference),
                    "calibration": len(role["calibration"]),
                    "evaluation": len(indices),
                    "latest_reference": int(timestamps[reference].max()),
                    "first_evaluation": int(timestamps[indices].min()),
                }
            ),
            flush=True,
        )
        scores, artifacts = {}, {}
        for modes in (1, 8):
            support = ConditionalMahalanobis.fit(matrices["train"], max_components=modes)
            scores[f"raw_mahalanobis_m{modes}"] = {
                name: support.score(values)[0].numpy()
                for name, values in matrices.items()
                if name != "train"
            }
            artifacts[f"raw_mahalanobis_m{modes}"] = support.state_dict()
        raw_reference = matrices["train"][: min(10000, len(reference))]
        scores["raw_knn"] = {
            name: knn_scores(raw_reference, values).numpy()
            for name, values in matrices.items()
            if name != "train"
        }
        deep_channels = {1: [], 8: []}
        seeds_artifact = []
        for seed in args.seeds:
            torch.manual_seed(seed)
            model = DependencyEnergy(features.shape[1])
            history = train(
                model,
                matrices["train"],
                matrices["selection"],
                corruption_groups(data["feature_names"]),
                seed=seed,
                epochs=args.epochs,
            )
            with torch.no_grad():
                embedded = {name: model.layers[:-1](values) for name, values in matrices.items()}
            emean = embedded["train"].mean(dim=0)
            escale = embedded["train"].std(dim=0, unbiased=False).clamp_min(1e-6)
            embedded = {name: (values - emean) / escale for name, values in embedded.items()}
            seed_artifact = {
                "seed": seed,
                "model": model.state_dict(),
                "embedding_mean": emean,
                "embedding_scale": escale,
                "history": history,
                "supports": {},
            }
            for modes in (1, 8):
                support = ConditionalMahalanobis.fit(embedded["train"], max_components=modes)
                energy = {
                    name: support.score(values)[0].numpy()
                    for name, values in embedded.items()
                    if name != "train"
                }
                median, q95 = np.quantile(energy["selection"], [0.5, 0.95])
                score_scale = max(float(q95 - median), 1e-6)
                deep_channels[modes].append(
                    {name: (value - median) / score_scale + 1 for name, value in energy.items()}
                )
                seed_artifact["supports"][str(modes)] = {
                    "geometry": support.state_dict(),
                    "score_center": float(median),
                    "score_scale": score_scale,
                }
            seeds_artifact.append(seed_artifact)
        for modes in (1, 8):
            scores[f"deep_mahalanobis_m{modes}_ensemble"] = {
                name: np.mean([channel[name] for channel in deep_channels[modes]], axis=0)
                for name in deep_channels[modes][0]
            }
        fusion_scales = {}
        for modes in (1, 8):
            raw, deep = scores["raw_knn"], scores[f"deep_mahalanobis_m{modes}_ensemble"]
            rscale = max(float(np.quantile(raw["selection"], 0.95)), 1e-6)
            dscale = max(float(np.quantile(deep["selection"], 0.95)), 1e-6)
            scores[f"raw_plus_deep_m{modes}"] = {
                name: np.maximum(raw[name] / rscale, deep[name] / dscale) for name in raw
            }
            fusion_scales[str(modes)] = {"raw": rscale, "deep": dscale}
        target_position = np.asarray([position[index] for index in indices])
        for method, arrays in scores.items():
            flags, valid = replay(
                arrays["evaluation"],
                labels[indices],
                captures[indices],
                timestamps[indices],
                arrays["calibration"],
                captures[role["calibration"]],
                timestamps[role["calibration"]],
                args.buffer,
            )
            if method not in all_scores:
                all_scores[method] = np.zeros(len(all_evaluation))
                all_flags[method] = np.zeros(len(all_evaluation), dtype=bool)
                all_valid[method] = np.zeros(len(all_evaluation), dtype=bool)
            all_scores[method][target_position] = arrays["evaluation"]
            all_flags[method][target_position] = flags
            all_valid[method][target_position] = valid
        artifact = {
            "format_version": 2,
            "knn_dtype": "float64",
            "input_mean": mean,
            "input_scale": scale,
            "feature_names": data["feature_names"].tolist(),
            "seed_models": seeds_artifact,
            "raw_reference": raw_reference,
            "raw_supports": artifacts,
            "fusion_scales": fusion_scales,
            "latest_reference_observed_at": int(timestamps[reference].max()),
        }
        torch.save(artifact, output / f"day_{capture}_detector.pt")
        completed[str(capture)] = {
            "reference": len(reference),
            "calibration": len(role["calibration"]),
            "evaluation": len(indices),
            "future_reference_overlap": False,
        }
        write_json(output / f"day_{capture}_complete.json", completed[str(capture)])
    truth, caps, categories = (
        labels[all_evaluation],
        captures[all_evaluation],
        data["type"][all_evaluation],
    )
    results = {}
    for method, values in all_scores.items():
        valid, flags = all_valid[method], all_flags[method]
        result = {
            "coverage": float(valid.mean()),
            "records": int(valid.sum()),
            "tpr": float(flags[valid & (truth == 1)].mean()),
            "fpr": float(flags[valid & (truth == 0)].mean()),
            "auroc": float(roc_auc_score(truth[valid], values[valid])),
            "auprc": float(average_precision_score(truth[valid], values[valid])),
        }
        result["per_capture"] = {
            str(int(capture)): {
                "normal": int(np.sum(valid & (caps == capture) & (truth == 0))),
                "attack": int(np.sum(valid & (caps == capture) & (truth == 1))),
                "fpr": float(flags[valid & (caps == capture) & (truth == 0)].mean())
                if np.any(valid & (caps == capture) & (truth == 0))
                else None,
                "tpr": float(flags[valid & (caps == capture) & (truth == 1)].mean())
                if np.any(valid & (caps == capture) & (truth == 1))
                else None,
            }
            for capture in sorted(set(caps))
        }
        result["per_type_tpr"] = {
            name: {
                "records": int(np.sum(valid & (truth == 1) & (categories == name))),
                "tpr": float(flags[valid & (truth == 1) & (categories == name)].mean()),
            }
            for name in sorted(set(categories[valid & (truth == 1)]))
        }
        results[method] = result
        print(
            json.dumps(
                {
                    "method": method,
                    "metrics": {key: result[key] for key in ("auroc", "tpr", "fpr", "coverage")},
                }
            ),
            flush=True,
        )
    uncertainty = {
        name: cluster_difference(
            (all_flags[name], all_valid[name]),
            (all_flags["raw_knn"], all_valid["raw_knn"]),
            truth,
            caps,
        )
        for name in ("deep_mahalanobis_m8_ensemble", "raw_plus_deep_m8")
    }
    write_json(
        output / "report.json",
        {
            "status": "chronologically_ordered_connection_development",
            "dataset": dataset,
            "days": completed,
            "methods": results,
            "capture_bootstrap_vs_knn": uncertainty,
        },
    )
    torch.save(
        {
            "scores": {name: torch.from_numpy(value) for name, value in all_scores.items()},
            "flags": {name: torch.from_numpy(value) for name, value in all_flags.items()},
            "valid": {name: torch.from_numpy(value) for name, value in all_valid.items()},
            "evaluation_indices": torch.from_numpy(all_evaluation),
        },
        output / "scores.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 41])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-evaluation", type=int, default=200000)
    parser.add_argument("--max-reference", type=int, default=20000)
    parser.add_argument("--buffer", type=int, default=200)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
