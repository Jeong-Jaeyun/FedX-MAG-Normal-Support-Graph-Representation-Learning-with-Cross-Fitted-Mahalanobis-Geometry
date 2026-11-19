"""Early verified-normal commissioning with model-aware held-out recalibration."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .adaptive_replay import replay_adaptive
from .commission_roles import early_normal_plans, minute_groups, normal_update_rows
from .daily_detector import DailyDeepEnergyDetector
from .experiments import sha256, write_json
from .record_calibration import cluster_difference
from .record_prequential import build_source, score_methods
from .window_bridge import aggregate_connections


def metrics(decision, labels, kinds):
    valid, flags = decision["valid"], decision["flags"]
    normal, attack = valid & (labels == 0), valid & (labels == 1)
    result = {
        "coverage": float(valid.mean()),
        "evaluated_windows": int(valid.sum()),
        "normal_windows": int(normal.sum()),
        "attack_windows": int(attack.sum()),
        "tpr": float(flags[attack].mean()) if attack.any() else None,
        "fpr": float(flags[normal].mean()) if normal.any() else None,
        "end_to_end_attack_recall": float(flags[labels == 1].mean()),
        "calibration_rank_auroc": float(roc_auc_score(labels[valid], -decision["pvalue"][valid]))
        if normal.any() and attack.any()
        else None,
        "per_type_tpr": {},
    }
    for kind in sorted({name for group in kinds for name in group}):
        member = np.array([kind in group for group in kinds])
        chosen = member & valid
        result["per_type_tpr"][kind] = {
            "available_windows": int(member.sum()),
            "evaluated_windows": int(chosen.sum()),
            "tpr": float(flags[chosen].mean()) if chosen.any() else None,
        }
    return result


def aggregate_scores(row_scores, indices, selection, groups, queries, dimension, aggregation):
    mapping = np.full(dimension, -1, dtype=np.int64)
    mapping[indices] = np.arange(len(indices))
    references = {name: values[mapping[selection]] for name, values in row_scores.items()}
    output = {}
    for name, values in row_scores.items():
        reference = references[name]
        scores = []
        for minute in queries:
            if int(minute) not in groups:
                scores.append(0.0)
                continue
            positions = mapping[groups[int(minute)]]
            if np.any(positions < 0):
                raise ValueError("Commissioning partially scored a complete minute")
            scores.append(aggregate_connections(values[positions], reference, aggregation))
        output[name] = np.array(scores)
    return output, references


@torch.no_grad()
def cached_calibration(
    detector, features, selection, calibration_rows, groups, source_minutes, aggregation
):
    indices = np.unique(np.r_[selection, calibration_rows])
    row_scores = score_methods(detector, features[indices])
    calibration, references = aggregate_scores(
        row_scores, indices, selection, groups, source_minutes, len(features), aggregation
    )
    return calibration, references


@torch.no_grad()
def cached_target(detector, features, row_minutes, target_minutes, references, groups, artifact):
    indices = np.flatnonzero(np.isin(row_minutes, target_minutes))
    print("commissioned target scoring rows", len(indices), flush=True)
    values = score_methods(detector, features[indices])
    audit_rows = np.linspace(0, len(indices) - 1, 128, dtype=np.int64)
    reproduced = score_methods(
        DailyDeepEnergyDetector.load(artifact), features[indices[audit_rows]]
    )
    audit = {}
    for name, scores in values.items():
        expected = scores[audit_rows]
        error = float(np.max(np.abs(reproduced[name] - expected) / (1 + np.abs(expected))))
        if error > 1e-5:
            raise ValueError(f"Restored commissioned score differs: {name}/{error}")
        audit[name] = error
    mapping = np.full(len(features), -1, dtype=np.int64)
    mapping[indices] = np.arange(len(indices))
    output = {}
    for name, scores in values.items():
        output[name] = np.array(
            [
                aggregate_connections(
                    scores[mapping[groups[int(t)]]], references[name], "rank_scan"
                )
                if int(t) in groups
                else 0.0
                for t in target_minutes
            ]
        )
    return output, audit


def run(args):
    torch.set_num_threads(args.workers)
    path, artifact, control, output = map(
        lambda value: Path(value).resolve(), (args.data, args.detector, args.control, args.output)
    )
    base = torch.load(artifact, weights_only=True)
    origin = json.loads((control / "registration.json").read_text())
    data_hash = sha256(path)
    if data_hash != base["source_data_sha256"] or data_hash != origin["data_sha256"]:
        raise ValueError("Commissioning data do not match their frozen controls")
    if origin["detector_sha256"] != sha256(artifact) or origin["aggregation"] != "rank_scan":
        raise ValueError("Commissioning requires the exact rank-scan frozen source control")
    with np.load(path) as payload:
        data = {key: payload[key] for key in payload.files}
    if str(data["dataset"]) != "cic":
        raise ValueError("This preregistered early adaptation run is CIC only")
    stored = torch.load(control / "scores.pt", weights_only=True)
    minutes, labels = stored["minutes"].numpy(), stored["labels"].numpy()
    features = torch.from_numpy(data["features"])
    groups = minute_groups(data["observed_at"])
    roles = {key: ids.numpy() for key, ids in base["source_role_indices"].items()}
    if any(
        np.any(data["label"][ids] != 0) or data["observed_at"][ids].max() >= minutes.min() * 60
        for ids in roles.values()
    ):
        raise ValueError("Original source roles must be normal and strictly before target")
    plans = early_normal_plans(data, minutes, labels, count=5, gap=300, delay=120)
    if not plans:
        raise ValueError("No capture supplies separated early normal feedback roles")
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": "cic",
            "data_sha256": data_hash,
            "base_encoder_sha256": sha256(artifact),
            "control_scores_sha256": sha256(control / "scores.pt"),
            "census_sha256": origin["census_sha256"],
            "feedback": (
                "whole-normal oracle AFTER prediction; missing confirmation is not an attack label"
            ),
            "prefix": (
                "first5 normal fit windows,300s raw-start purge,first5 normal selection windows"
            ),
            "references": "20000 total,up to5000 most recent already verified target-normal rows",
            "selection": "all original selection normals plus early held-out target normals",
            "fine_tune": "5epochs dependency BCE,initial checkpoint,normal validation only",
            "input_units": "inherit original source scaler in every branch",
            "readiness": "selection feedback end +max(120s,fit/calibration compute time)",
            "calibration": (
                "rescore independent source windows; append only whole normals "
                "whose raw starts exceed new info end+300s"
            ),
            "query_scope": "ALL target windows,old model until strict readiness,none dropped",
            "comparison": "reference-only / fine-tuned / same-reset frozen / original frozen",
        },
    )
    source_minutes = stored["methods"]["raw_knn"]["source_calibration_minutes"].numpy()
    if any(not np.isin(groups[int(t)], roles["calibration"]).all() for t in source_minutes):
        raise ValueError("A source calibration window mixes reserved roles")
    seeds = [item["seed"] for item in base["seed_models"]]
    stage_cache, previous_tuned, stage_metadata = {"reference": [], "tuned": []}, base, []
    for plan in plans:
        reference, selection = normal_update_rows(data, roles, groups, minutes, labels, plan)
        results, runtimes, states, audits = {}, {}, {}, {}
        for branch, epochs, initial in (("reference", 0, base), ("tuned", 5, previous_tuned)):
            print(
                "FIT early commission",
                plan["day"],
                branch,
                "reference",
                len(reference),
                "selection",
                len(selection),
                flush=True,
            )
            start = time.perf_counter()
            # Training is intentionally outside no_grad; only normal rows enter.
            state = build_source(
                data,
                reference,
                selection,
                argparse.Namespace(
                    max_reference=20000, seeds=seeds, epochs=epochs, objective="dependency"
                ),
                initial_state=initial,
            )
            state["source_data_sha256"] = data_hash
            state["source_role_indices"] = {
                "train": torch.from_numpy(reference.copy()),
                "selection": torch.from_numpy(selection.copy()),
                "calibration": torch.from_numpy(roles["calibration"].copy()),
            }
            state["commission_info_end"] = plan["selection_observation_end"]
            if state["latest_reference_observed_at"] > plan["reference_observation_end"]:
                raise ValueError("A commissioned fit used future reference feedback")
            model_path = output / "models" / f"day_{plan['day']}_{branch}.pt"
            torch.save(state, model_path)
            detector = DailyDeepEnergyDetector.load(model_path)
            calibration, rank_reference = cached_calibration(
                detector,
                features,
                selection,
                roles["calibration"],
                groups,
                source_minutes,
                "rank_scan",
            )
            runtimes[branch] = time.perf_counter() - start
            scores, audit = cached_target(
                detector,
                features,
                data["observed_at"] // 60,
                minutes,
                rank_reference,
                groups,
                model_path,
            )
            results[branch] = {"energy": scores, "calibration": calibration}
            states[branch], audits[branch] = state, audit
        previous_tuned = states["tuned"]
        ready = plan["feedback_available_at"] + max(120, math.ceil(max(runtimes.values())))
        allowed = np.array(
            [
                int(t) in groups
                and data["timestamp"][groups[int(t)]].min() > plan["calibration_raw_start_after"]
                for t in minutes
            ]
        )
        for name in ("raw_knn", "raw_mahalanobis_m1", "raw_mahalanobis_m8"):
            for key in ("energy", "calibration"):
                a, b = (results[branch][key][name] for branch in ("reference", "tuned"))
                if float(np.max(np.abs(a - b) / (1 + np.abs(a)))) > 1e-8:
                    raise ValueError(f"Commissioning raw controls differ: {name}/{key}")
        for branch in results:
            stage_cache[branch].append({**results[branch], "ready_at": ready, "admit": allowed})
        stage_metadata.append(
            {
                "day": plan["day"],
                "train_minutes": plan["train_minutes"],
                "selection_minutes": plan["selection_minutes"],
                "ready_at": ready,
                "reference_observation_end": plan["reference_observation_end"],
                "selection_observation_end": plan["selection_observation_end"],
                "reference_rows": len(reference),
                "selection_rows": len(selection),
                "target_reference_rows": int(
                    np.sum(data["observed_at"][reference] >= minutes.min() * 60)
                ),
                "normal_fit_and_calibration_seconds": runtimes,
                "restore_scaled_errors": audits,
                "artifacts": {
                    branch: {
                        "path": f"models/day_{plan['day']}_{branch}.pt",
                        "sha256": sha256(output / "models" / f"day_{plan['day']}_{branch}.pt"),
                    }
                    for branch in results
                },
                "raw_control_pairs_exact": True,
            }
        )
        print("READY early commission", stage_metadata[-1], flush=True)
    decisions, reports, method_specs = {}, {}, []
    for name, item in stored["methods"].items():
        decision = replay_adaptive(
            minutes,
            labels,
            item["observed"].numpy(),
            item["energy"].numpy(),
            item["source_calibration"].numpy(),
            [],
        )
        if not np.array_equal(decision["flags"], item["flags"].numpy()) or not np.array_equal(
            decision["valid"], item["valid"].numpy()
        ):
            raise ValueError(f"Original frozen decision replay changed: {name}")
        decisions[name] = decision
    core = [name for name in stage_cache["reference"][0]["energy"]]
    for name in core:
        if name.startswith("raw_") and not name.startswith("raw_plus_"):
            method_specs.append((f"commissioned_{name}", name, "reference"))
        else:
            method_specs.extend(
                (f"{branch}_{name}", name, branch) for branch in ("reference", "tuned")
            )
        if not name.startswith("raw_plus_"):
            method_specs.append((f"same_reset_{name}", name, "reset"))
    for output_name, name, branch in method_specs:
        original = stored["methods"][name]
        stages = [
            {
                "ready_at": stage["ready_at"],
                "admit": stage["admit"],
                "energy": original["energy"].numpy()
                if branch == "reset"
                else stage["energy"][name],
                "calibration": original["source_calibration"].numpy()
                if branch == "reset"
                else stage["calibration"][name],
            }
            for stage in stage_cache["reference" if branch == "reset" else branch]
        ]
        decisions[output_name] = replay_adaptive(
            minutes,
            labels,
            original["observed"].numpy(),
            original["energy"].numpy(),
            original["source_calibration"].numpy(),
            stages,
        )
    for name, decision in decisions.items():
        reports[name] = metrics(decision, labels, stored["types"])
        print(name, {k: reports[name][k] for k in ("tpr", "fpr", "coverage")}, flush=True)
    baseline = decisions["deep_mahalanobis_m1_ensemble"]
    differences = {
        name: cluster_difference(
            (value["flags"], value["valid"]),
            (baseline["flags"], baseline["valid"]),
            labels,
            minutes // 60,
        )
        for name, value in decisions.items()
        if name.startswith(("reference_", "tuned_"))
    }
    torch.save(
        {
            name: {key: torch.from_numpy(value) for key, value in item.items()}
            for name, item in decisions.items()
        },
        output / "decisions.pt",
    )
    torch.save(
        {
            "stages": {
                branch: [
                    {
                        "ready_at": stage["ready_at"],
                        "admit": torch.from_numpy(stage["admit"]),
                        "energy": {
                            name: torch.from_numpy(value) for name, value in stage["energy"].items()
                        },
                        "calibration": {
                            name: torch.from_numpy(value)
                            for name, value in stage["calibration"].items()
                        },
                    }
                    for stage in stages
                ]
                for branch, stages in stage_cache.items()
            },
            "method_specs": method_specs,
        },
        output / "adaptive_scores.pt",
    )
    write_json(output / "model_timeline.json", {"stages": stage_metadata})
    write_json(
        output / "report.json",
        {
            "status": "early_verified_normal_commission_prequential_development",
            "dataset": "cic",
            "methods": reports,
            "hour_bootstrap_vs_frozen_deep1": differences,
            "all_original_control_decisions_exact": True,
            "normal_feedback_after_prediction": True,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
