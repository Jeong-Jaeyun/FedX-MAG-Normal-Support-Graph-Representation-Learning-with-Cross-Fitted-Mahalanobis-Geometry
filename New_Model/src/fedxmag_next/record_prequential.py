"""Evaluate every target connection, including the adaptation/warm-up period.

Before today's adapted model is ready, use the latest previously available
model. Actual known-normal feedback arrives only after predictions. The warm-up
attacks are never dropped to make suffix-only detection look easier.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .context_association import cohort_ids, train_association
from .contrastive_energy import DependencyEnergy, train
from .daily_detector import DailyDeepEnergyDetector
from .experiments import knn_scores, sha256, write_json
from .geometry import ConditionalMahalanobis
from .geometry_training import GeometryResidualEncoder, train_geometry
from .neural_transforms import TransformationEnergy, train_transformations
from .record_calibration import cluster_difference, replay
from .record_experiment import corruption_groups


def build_source(
    data: dict,
    source: np.ndarray,
    selection: np.ndarray,
    args: argparse.Namespace,
    *,
    initial_state: dict | None = None,
) -> dict:
    if args.epochs < 0 or (args.epochs == 0 and initial_state is None):
        raise ValueError("Zero epochs require an explicit frozen initial encoder")
    if initial_state is not None and (
        initial_state.get("encoder_kind", "dependency") != "dependency"
        or getattr(args, "objective", "dependency") != "dependency"
        or tuple(initial_state["feature_names"]) != tuple(data["feature_names"])
    ):
        raise ValueError("Warm adaptation requires an identical dependency encoder schema")
    if len(source) > args.max_reference:
        source = source[np.argsort(data["signature"][source])[: args.max_reference]]
    features = torch.from_numpy(data["features"])
    normal = features[source]
    if initial_state is None:
        mean, scale = normal.mean(dim=0), normal.std(dim=0, unbiased=False).clamp_min(0.05)
    else:
        mean, scale = initial_state["input_mean"].clone(), initial_state["input_scale"].clone()
    reference, selected = (normal - mean) / scale, (features[selection] - mean) / scale
    keep = initial_state.get("input_keep") if initial_state is not None else None
    if initial_state is not None and keep is not None:
        keep = keep.clone()
        reference, selected = reference * keep, selected * keep
    elif initial_state is None and getattr(args, "constant_feature_policy", "preserve") == "ignore":
        keep = normal.std(dim=0, unbiased=False) > 1e-6
        reference, selected = reference * keep, selected * keep
    raw_reference = reference[: min(10000, len(reference))]
    raw_selected = knn_scores(raw_reference, selected)
    raw_scale = max(float(torch.quantile(raw_selected, 0.95)), 1e-6)
    encoder_kind = getattr(args, "objective", "dependency")
    if encoder_kind not in (
        "dependency",
        "neural_transformations",
        "association",
        "geometry_margin",
    ):
        raise ValueError("Unknown source encoder objective")
    transformation_config = {"transforms": 4, "temperature": 0.1}
    state = {
        "format_version": 2,
        "knn_dtype": "float64",
        "input_mean": mean,
        "input_scale": scale,
        "input_keep": keep,
        "encoder_kind": encoder_kind,
        "warm_started": initial_state is not None,
        "transformation_config": transformation_config,
        "association_context_columns": [
            i for i, name in enumerate(data["feature_names"]) if name.startswith("context_")
        ],
        "feature_names": data["feature_names"].tolist(),
        "raw_reference": raw_reference,
        "seed_models": [],
        "raw_supports": {},
        "fusion_scales": {},
        "latest_reference_observed_at": int(data["observed_at"][source].max()),
    }
    selected_deep = {1: [], 8: []}
    for modes in (1, 8):
        state["raw_supports"][f"raw_mahalanobis_m{modes}"] = ConditionalMahalanobis.fit(
            reference, max_components=modes
        ).state_dict()
    for seed in args.seeds:
        torch.manual_seed(seed)
        if encoder_kind == "neural_transformations":
            model = TransformationEnergy(features.shape[1], **transformation_config)
            history = train_transformations(
                model, reference, selected, seed=seed, epochs=args.epochs
            )
        elif encoder_kind == "association":
            model = DependencyEnergy(features.shape[1])
            history = train_association(
                model,
                reference,
                selected,
                cohort_ids(normal),
                cohort_ids(features[selection]),
                state["association_context_columns"],
                seed=seed,
                epochs=args.epochs,
            )
        elif encoder_kind == "geometry_margin":
            model = GeometryResidualEncoder(features.shape[1])
            history = train_geometry(
                model,
                reference,
                selected,
                cohort_ids(normal),
                cohort_ids(features[selection]),
                state["association_context_columns"],
                seed=seed,
                epochs=args.epochs,
            )
        else:
            model = DependencyEnergy(features.shape[1])
            if initial_state is not None:
                previous = {item["seed"]: item for item in initial_state["seed_models"]}
                if seed not in previous:
                    raise ValueError("Warm adaptation requested an unknown initial seed")
                model.load_state_dict(previous[seed]["model"])
            history = (
                train(
                    model,
                    reference,
                    selected,
                    corruption_groups(data["feature_names"]),
                    seed=seed,
                    epochs=args.epochs,
                    include_initial=initial_state is not None,
                )
                if args.epochs
                else [{"epoch": 0, "neural_weights_frozen": True}]
            )
            model.eval()
        with torch.no_grad():
            embedded, embedded_selection = model.layers[:-1](reference), model.layers[:-1](selected)
        emean = embedded.mean(dim=0)
        escale = embedded.std(dim=0, unbiased=False).clamp_min(1e-6)
        if encoder_kind == "geometry_margin":
            # Raw coordinates were already standardized at input. Constant raw
            # fields must not acquire an artificial second1e-6 denominator.
            emean[: features.shape[1]] = 0
            escale[: features.shape[1]] = 1
        embedded, embedded_selection = (
            (embedded - emean) / escale,
            (embedded_selection - emean) / escale,
        )
        item = {
            "seed": seed,
            "model": model.state_dict(),
            "embedding_mean": emean,
            "embedding_scale": escale,
            "history": history,
            "supports": {},
        }
        if encoder_kind in ("neural_transformations", "association"):
            with torch.no_grad():
                native = torch.cat([model(x) for x in selected.split(8192)]).double()
            center, q95 = torch.quantile(native, torch.tensor([0.5, 0.95], dtype=torch.float64))
            item["native_score_center"] = float(center)
            item["native_score_scale"] = max(float(q95 - center), 1e-6)
        for modes in (1, 8):
            support = ConditionalMahalanobis.fit(embedded, max_components=modes)
            energy = support.score(embedded_selection)[0]
            median, q95 = torch.quantile(energy, torch.tensor([0.5, 0.95], dtype=torch.float64))
            score_scale = max(float(q95 - median), 1e-6)
            selected_deep[modes].append((energy - median) / score_scale + 1)
            item["supports"][str(modes)] = {
                "geometry": support.state_dict(),
                "score_center": float(median),
                "score_scale": score_scale,
            }
        state["seed_models"].append(item)
    for modes in (1, 8):
        deep = torch.stack(selected_deep[modes]).mean(dim=0)
        state["fusion_scales"][str(modes)] = {
            "raw": raw_scale,
            "deep": max(float(torch.quantile(deep, 0.95)), 1e-6),
        }
    return state


def score_methods(
    detector: DailyDeepEnergyDetector, features: torch.Tensor
) -> dict[str, np.ndarray]:
    result = {}
    for modes, score in detector.score_all(features).items():
        result[f"raw_mahalanobis_m{modes}"] = score["raw_mahalanobis"].numpy()
        result["raw_knn"] = score["raw_knn"].numpy()
        result[f"deep_mahalanobis_m{modes}_ensemble"] = score["deep_mahalanobis"].numpy()
        result[f"raw_plus_deep_m{modes}"] = score["hybrid"].numpy()
        if "transformation" in score:
            result["deep_transformation_native_ensemble"] = score["transformation"].numpy()
        if "association" in score:
            result["deep_association_native_ensemble"] = score["association"].numpy()
    return result


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    output, previous = Path(args.output).resolve(), Path(args.daily_run).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_path = Path(args.data).resolve()
    with np.load(data_path) as payload:
        data = {key: payload[key] for key in payload.files}
    previous_registration = json.loads((previous / "registration.json").read_text())
    if previous_registration["data_sha256"] != sha256(data_path):
        raise ValueError("Daily models and prequential observations differ")
    dataset, offset = str(data.get("dataset", "ton")), int(data.get("utc_offset_seconds", 36000))
    if dataset == "xiiotid":
        days = np.unique((data["timestamp"] + offset) // 86400)
        cutoff, end = (
            int(days[int(0.6 * len(days))]) * 86400 - offset,
            (int(days[-1]) + 1) * 86400 - offset,
        )
    else:
        dates = ("2019-04-23", "2019-05-01") if dataset == "ton" else ("2017-07-06", "2017-07-08")
        cutoff, end = [int(np.datetime64(day, "s").astype(np.int64)) - offset for day in dates]
    ts, labels, caps = data["observed_at"], data["label"], data["capture"]
    source = np.flatnonzero((ts < cutoff) & (labels == 0))
    source = source[np.argsort(ts[source], kind="stable")]
    a, b = int(0.7 * len(source)), int(0.85 * len(source))
    source_train, selection, source_cal = source[:a], source[a:b], source[b:]
    target = np.flatnonzero((ts >= cutoff) & (ts < end))
    # Retain the whole prepared target stream. Earlier fixed row sampling was
    # label independent; no further attack/type-specific removal occurs here.
    write_json(
        output / "registration.json",
        {
            "status": "full_target_prequential_development",
            "args": vars(args),
            "dataset": dataset,
            "data_sha256": sha256(data_path),
            "all_target_records": len(target),
            "includes_warmup_attacks": True,
            "calibration_fpr": args.calibration_fpr,
            "model_updates": "latest source/past-day model until today's adapted model is ready",
            "feedback": (
                "verified normals after scoring; "
                "held-out calibration after the model reference cutoff"
            ),
            "daily_registration_sha256": sha256(previous / "registration.json"),
        },
    )
    if args.source_detector:
        source_path = Path(args.source_detector).resolve()
        origin = json.loads((source_path.parent / "registration.json").read_text())
        if origin["data_sha256"] != sha256(data_path):
            raise ValueError("Reusable source model was fitted to another dataset")
        state = torch.load(source_path, map_location="cpu", weights_only=True)
    else:
        state = build_source(data, source_train, selection, args)
    detector = DailyDeepEnergyDetector(state)
    features = torch.from_numpy(data["features"])
    detector.upgrade_knn_precision(features[selection])
    state = detector.state
    torch.save(state, output / "source_detector.pt")
    current_model_id = -1
    position = {index: item for item, index in enumerate(target)}
    all_scores, all_flags, all_valid, model_used = {}, {}, {}, np.full(len(target), -1, dtype=int)
    phase_report = []
    phase_calibrations = []
    minimum_calibration = math.ceil(1 / args.calibration_fpr) - 1
    for capture in sorted(set(caps[target])):
        members = target[caps[target] == capture]
        members = members[np.argsort(ts[members], kind="stable")]
        first, ready = np.quantile(ts[members], [0.2, 0.4], method="lower")
        for phase, queries in (
            ("before_adaptation", members[ts[members] <= ready]),
            ("after_adaptation", members[ts[members] > ready]),
        ):
            if not len(queries):
                continue
            if phase == "after_adaptation":
                calibration = members[
                    (ts[members] > first) & (ts[members] <= ready) & (labels[members] == 0)
                ]
                if len(calibration) >= minimum_calibration:
                    detector = DailyDeepEnergyDetector.load(previous / f"day_{capture}_detector.pt")
                    detector.upgrade_knn_precision(features[selection])
                    current_model_id = int(capture)
                else:
                    eligible = np.concatenate((source_cal, target[labels[target] == 0]))
                    eligible = eligible[
                        (ts[eligible] > detector.state["latest_reference_observed_at"])
                        & (ts[eligible] < ts[queries].min())
                    ]
                    calibration = eligible[np.argsort(ts[eligible], kind="stable")][-args.buffer :]
            else:
                if not phase_report:
                    calibration = source_cal
                else:
                    minimum = detector.state["latest_reference_observed_at"]
                    eligible = np.concatenate((source_cal, target[labels[target] == 0]))
                    calibration = eligible[
                        (ts[eligible] > minimum) & (ts[eligible] < ts[queries].min())
                    ]
                    calibration = calibration[np.argsort(ts[calibration], kind="stable")][
                        -args.buffer :
                    ]
            if detector.state["latest_reference_observed_at"] >= int(ts[queries].min()):
                raise ValueError("A prequential model used a future reference")
            if len(calibration) and int(ts[calibration].max()) >= int(ts[queries].min()):
                raise ValueError("A prequential calibration used future feedback")
            print(
                json.dumps(
                    {
                        "phase": phase,
                        "capture": int(capture),
                        "records": len(queries),
                        "normal_calibration": len(calibration),
                    }
                ),
                flush=True,
            )
            combined = torch.cat((features[calibration], features[queries]))
            score = score_methods(detector, combined)
            selected_position = np.asarray([position[index] for index in queries])
            phase_cache = {
                "capture": int(capture),
                "phase": phase,
                "query_positions": torch.from_numpy(selected_position),
                "calibration_timestamps": torch.from_numpy(ts[calibration]),
                "methods": {},
            }
            for method, values in score.items():
                cal_values, actual = values[: len(calibration)], values[len(calibration) :]
                flags, valid = replay(
                    actual,
                    labels[queries],
                    caps[queries],
                    ts[queries],
                    cal_values,
                    np.full(len(calibration), capture),
                    ts[calibration],
                    args.buffer,
                    fpr=args.calibration_fpr,
                )
                phase_cache["methods"][method] = torch.from_numpy(cal_values.copy())
                if method not in all_scores:
                    all_scores[method] = np.zeros(len(target))
                    all_flags[method] = np.zeros(len(target), dtype=bool)
                    all_valid[method] = np.zeros(len(target), dtype=bool)
                all_scores[method][selected_position] = actual
                all_flags[method][selected_position] = flags
                all_valid[method][selected_position] = valid
            model_used[selected_position] = current_model_id
            phase_calibrations.append(phase_cache)
            phase_report.append(
                {
                    "capture": int(capture),
                    "phase": phase,
                    "records": len(queries),
                    "latest_model_reference": detector.state["latest_reference_observed_at"],
                    "first_query": int(ts[queries].min()),
                    "normal_calibration": len(calibration),
                }
            )
    truth, categories, capture = labels[target], data["type"][target], caps[target]
    results = {}
    for method, values in all_scores.items():
        flags, valid = all_flags[method], all_valid[method]
        result = {
            "coverage": float(valid.mean()),
            "records": int(valid.sum()),
            "auroc": float(roc_auc_score(truth[valid], values[valid])),
            "tpr": float(flags[valid & (truth == 1)].mean()),
            "fpr": float(flags[valid & (truth == 0)].mean()),
            "per_type_tpr": {
                name: {
                    "available_records": int(np.sum((truth == 1) & (categories == name))),
                    "evaluated_records": int(np.sum(valid & (truth == 1) & (categories == name))),
                    "tpr": float(flags[valid & (truth == 1) & (categories == name)].mean()),
                }
                for name in sorted(set(categories[truth == 1]))
            },
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
    intervals = {
        name: cluster_difference(
            (all_flags[name], all_valid[name]),
            (all_flags["raw_knn"], all_valid["raw_knn"]),
            truth,
            capture,
        )
        for name in ("deep_mahalanobis_m8_ensemble", "raw_plus_deep_m8")
    }
    write_json(
        output / "report.json",
        {
            "status": "full_target_prequential_development",
            "dataset": dataset,
            "phases": phase_report,
            "methods": results,
            "capture_bootstrap_vs_knn": intervals,
        },
    )
    torch.save(
        {
            "scores": {k: torch.from_numpy(v) for k, v in all_scores.items()},
            "flags": {k: torch.from_numpy(v) for k, v in all_flags.items()},
            "valid": {k: torch.from_numpy(v) for k, v in all_valid.items()},
            "target_indices": torch.from_numpy(target),
            "model_used": torch.from_numpy(model_used),
        },
        output / "scores.pt",
    )
    torch.save(phase_calibrations, output / "calibrations.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--daily-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 41])
    parser.add_argument("--max-reference", type=int, default=20000)
    parser.add_argument("--buffer", type=int, default=200)
    parser.add_argument("--source-detector", help="Reuse a source-only model bound to this dataset")
    parser.add_argument("--calibration-fpr", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.calibration_fpr < 1:
        parser.error("calibration-fpr must lie in (0, 1)")
    if args.buffer < math.ceil(1 / args.calibration_fpr) - 1:
        parser.error("normal buffer cannot support the requested calibration FPR")
    run(args)


if __name__ == "__main__":
    main()
