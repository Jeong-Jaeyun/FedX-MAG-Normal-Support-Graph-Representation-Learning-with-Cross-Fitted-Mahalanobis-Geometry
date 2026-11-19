"""Connection-level normal-only energy experiment with matched raw baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .contrastive_energy import DependencyEnergy, train
from .detector import conformal_threshold
from .experiments import knn_scores, sha256, write_json
from .geometry import ConditionalMahalanobis
from .record_data import FEATURES, NUMERIC, PORTS, PROTOCOLS, SERVICES, STATES


def corruption_groups(feature_names=None) -> list[list[int]]:
    position = {name: index for index, name in enumerate(FEATURES)}
    groups = [[position[f"log_{name}"], position[f"available_{name}"]] for name in NUMERIC]
    for prefix, names in (("protocol", PROTOCOLS), ("service", SERVICES), ("state", STATES)):
        groups.append([position[f"{prefix}_{name}"] for name in (*names, "other")])
    groups.append(
        [
            position[name]
            for name in (
                "log_dst_port",
                "source_privileged_port",
                "destination_privileged_port",
                *[f"destination_port_{port}" for port in PORTS],
                "destination_port_other",
                "available_src_port",
                "available_dst_port",
            )
        ]
    )
    groups.append(
        [
            position[name]
            for name in (
                "log_http_uri_length",
                "uri_percent_fraction",
                "uri_punctuation_fraction",
                "available_http_uri",
            )
        ]
    )
    names = tuple(FEATURES if feature_names is None else feature_names)
    if names[: len(FEATURES)] != FEATURES:
        raise ValueError("Common connection schema prefix differs")
    # HTTP summaries include deterministic versions of URI measurements. Corrupt
    # their entire provenance group together, never one copy without the other.
    contextual = []
    for index in range(len(FEATURES), len(names)):
        if names[index].startswith("context_"):
            contextual.append(index)
        elif names[index].startswith("payload_"):
            groups[-1].append(index)
        elif names[index].startswith("control_"):
            # ICMP pseudo-ports and transport source ports share the existing
            # destination/source-port provenance group.
            groups[-2].append(index)
        else:
            raise ValueError(f"Unregistered derived feature provenance: {names[index]}")
    if contextual:
        # Trailing aggregates share their historical event set. Permute them
        # together so the synthetic task cannot exploit copied counts/ratios.
        groups.append(contextual)
    if sorted(index for group in groups for index in group) != list(range(len(names))):
        raise ValueError("Every record feature must belong to exactly one corruption group")
    return groups


def evaluate(
    values: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    captures: np.ndarray,
    thresholds: dict[int, float | None],
) -> dict:
    valid = np.asarray([thresholds.get(int(capture)) is not None for capture in captures])
    if not valid.any() or len(set(labels[valid])) != 2:
        return {"status": "insufficient_calibration_coverage", "coverage": float(valid.mean())}
    scores, truth, caps, types = values[valid], labels[valid], captures[valid], categories[valid]
    flags = scores > np.asarray([thresholds[int(capture)] for capture in caps])
    return {
        "status": "evaluated",
        "coverage": float(valid.mean()),
        "records": int(valid.sum()),
        "normal": int(np.sum(truth == 0)),
        "attack": int(np.sum(truth == 1)),
        "auroc": float(roc_auc_score(truth, scores)),
        "auprc": float(average_precision_score(truth, scores)),
        "tpr": float(flags[truth == 1].mean()),
        "fpr": float(flags[truth == 0].mean()),
        "per_capture": {
            str(int(capture)): {
                "normal": int(np.sum((caps == capture) & (truth == 0))),
                "attack": int(np.sum((caps == capture) & (truth == 1))),
                "fpr": float(flags[(caps == capture) & (truth == 0)].mean())
                if np.any((caps == capture) & (truth == 0))
                else None,
                "tpr": float(flags[(caps == capture) & (truth == 1)].mean())
                if np.any((caps == capture) & (truth == 1))
                else None,
            }
            for capture in sorted(set(caps))
        },
        "per_type_tpr": {
            name: {
                "records": int(np.sum((types == name) & (truth == 1))),
                "tpr": float(flags[(types == name) & (truth == 1)].mean()),
            }
            for name in sorted(set(types[truth == 1]))
        },
    }


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_path = Path(args.data).resolve()
    with np.load(data_path) as payload:
        data = {key: payload[key] for key in payload.files}
    if tuple(data["feature_names"][: len(FEATURES)]) != FEATURES:
        raise ValueError("Record feature schema differs")
    timestamp, labels, captures = data["observed_at"], data["label"], data["capture"]
    dataset = str(data.get("dataset", "ton"))
    offset = int(data.get("utc_offset_seconds", 10 * 3600))
    start_day, end_day = (
        ("2019-04-23", "2019-05-01") if dataset == "ton" else ("2017-07-06", "2017-07-08")
    )
    cutoff = int(np.datetime64(start_day, "s").astype(np.int64)) - offset
    end = int(np.datetime64(end_day, "s").astype(np.int64)) - offset
    source_normal = np.flatnonzero((timestamp < cutoff) & (labels == 0))
    source_normal = source_normal[np.argsort(timestamp[source_normal], kind="stable")]
    split1, split2 = int(0.7 * len(source_normal)), int(0.85 * len(source_normal))
    source_train, selection, source_cal = (
        source_normal[:split1],
        source_normal[split1:split2],
        source_normal[split2:],
    )
    target = np.flatnonzero((timestamp >= cutoff) & (timestamp < end))
    reference, calibration, evaluation = [], [], []
    for capture in sorted(set(captures[target])):
        members = target[captures[target] == capture]
        ordered = members[np.argsort(timestamp[members], kind="stable")]
        # Fixed time cutoffs preserve equal-time groups across the role boundary.
        first = np.quantile(timestamp[ordered], 0.2, method="lower")
        second = np.quantile(timestamp[ordered], 0.4, method="lower")
        reference.extend(ordered[(timestamp[ordered] <= first) & (labels[ordered] == 0)].tolist())
        calibration.extend(
            ordered[
                (timestamp[ordered] > first)
                & (timestamp[ordered] <= second)
                & (labels[ordered] == 0)
            ].tolist()
        )
        evaluation.extend(ordered[timestamp[ordered] > second].tolist())
    reference, calibration, evaluation = [
        np.asarray(value, dtype=int) for value in (reference, calibration, evaluation)
    ]
    # Bound runtime using observation signatures, never labels or anomaly scores.
    if len(evaluation) > args.max_evaluation:
        order = np.argsort(data["signature"][evaluation])[: args.max_evaluation]
        evaluation = evaluation[order]
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "status": "actual_connection_level_development",
            "data_sha256": sha256(data_path),
            "fit_labels": "normal-only; synthetic negatives only",
            "source": (
                f"observed_at before {start_day}, UTC offset {offset} seconds; "
                "normal train/selection/calibration chronological"
            ),
            "target": (
                f"{start_day} until {end_day} exclusive; per-day time quantiles "
                "20% reference / 20% calibration / 60% evaluation"
            ),
            "causality": (
                "Use record completion time; target labels supply trusted normal prefixes "
                "and evaluate the suffix"
            ),
            "counts": {
                "source_train": len(source_train),
                "selection": len(selection),
                "source_calibration": len(source_cal),
                "target_reference": len(reference),
                "target_calibration": len(calibration),
                "evaluation": len(evaluation),
            },
            "seeds": args.seeds,
        },
    )
    print(
        json.dumps(
            {
                "phase": "record_roles",
                "source_train": len(source_train),
                "target_reference": len(reference),
                "target_calibration": len(calibration),
                "evaluation": len(evaluation),
            }
        ),
        flush=True,
    )
    features = torch.from_numpy(data["features"])
    all_results, all_scores = {}, {}
    for arm in args.arms:
        training = source_train if arm == "source" else np.concatenate((source_train, reference))
        normal = features[training]
        mean, scale = normal.mean(dim=0), normal.std(dim=0, unbiased=False).clamp_min(0.05)
        roles = {
            "train": training,
            "selection": selection,
            "calibration": source_cal if arm == "source" else calibration,
            "evaluation": evaluation,
        }
        matrices = {role: (features[indices] - mean) / scale for role, indices in roles.items()}
        reference_rows = matrices["train"]
        if len(reference_rows) > 10000:
            order = torch.randperm(
                len(reference_rows), generator=torch.Generator().manual_seed(91)
            )[:10000]
            knn_reference = reference_rows[order]
        else:
            knn_reference = reference_rows
        scores = {}
        for modes in (1, 8):
            support = ConditionalMahalanobis.fit(reference_rows, max_components=modes)
            scores[f"raw_mahalanobis_m{modes}"] = {
                role: support.score(values)[0].numpy()
                for role, values in matrices.items()
                if role != "train"
            }
        scores["raw_knn"] = {
            role: knn_scores(knn_reference, values).numpy()
            for role, values in matrices.items()
            if role != "train"
        }
        deep_channels = {1: [], 8: []}
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
                embedded = {role: model.layers[:-1](values) for role, values in matrices.items()}
            emean = embedded["train"].mean(dim=0)
            escale = embedded["train"].std(dim=0, unbiased=False).clamp_min(1e-6)
            embedded = {role: (values - emean) / escale for role, values in embedded.items()}
            for modes in (1, 8):
                support = ConditionalMahalanobis.fit(embedded["train"], max_components=modes)
                values = {
                    role: support.score(embedding)[0].numpy()
                    for role, embedding in embedded.items()
                    if role != "train"
                }
                median, q95 = np.quantile(values["selection"], [0.5, 0.95])
                deep_channels[modes].append(
                    {
                        role: (array - median) / max(q95 - median, 1e-6) + 1
                        for role, array in values.items()
                    }
                )
            torch.save(
                {"model": model.state_dict(), "mean": mean, "scale": scale, "history": history},
                output / f"{arm}_seed{seed}.pt",
            )
        for modes in (1, 8):
            scores[f"deep_mahalanobis_m{modes}_ensemble"] = {
                role: np.mean([channel[role] for channel in deep_channels[modes]], axis=0)
                for role in deep_channels[modes][0]
            }
        for modes in (1, 8):
            baseline, deep = scores["raw_knn"], scores[f"deep_mahalanobis_m{modes}_ensemble"]
            bscale, dscale = (
                max(float(np.quantile(baseline["selection"], 0.95)), 1e-6),
                max(float(np.quantile(deep["selection"], 0.95)), 1e-6),
            )
            scores[f"raw_plus_deep_m{modes}"] = {
                role: np.maximum(baseline[role] / bscale, deep[role] / dscale) for role in baseline
            }
        for method, values in scores.items():
            if arm == "source":
                thresholds = {
                    int(capture): conformal_threshold(values["calibration"], 0.05)
                    for capture in set(captures[evaluation])
                }
            else:
                thresholds = {
                    int(capture): conformal_threshold(
                        values["calibration"][captures[calibration] == capture], 0.05
                    )
                    for capture in set(captures[evaluation])
                }
            result = evaluate(
                values["evaluation"],
                labels[evaluation],
                data["type"][evaluation],
                captures[evaluation],
                thresholds,
            )
            all_results[f"{arm}_{method}"] = result
            print(
                json.dumps(
                    {
                        "method": f"{arm}_{method}",
                        "metrics": {
                            key: result.get(key) for key in ("auroc", "tpr", "fpr", "coverage")
                        },
                    }
                ),
                flush=True,
            )
        all_scores[arm] = {
            name: {role: torch.from_numpy(value) for role, value in values.items()}
            for name, values in scores.items()
        }
        write_json(
            output / f"{arm}_report.json",
            {"methods": {key: value for key, value in all_results.items() if key.startswith(arm)}},
        )
    write_json(
        output / "report.json",
        {"status": "actual_connection_level_development", "methods": all_results},
    )
    torch.save(
        {"scores": all_scores, "evaluation_indices": torch.from_numpy(evaluation)},
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
    parser.add_argument(
        "--arms", nargs="+", choices=("source", "commissioned"), default=["source", "commissioned"]
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
