"""Source-normal score units around unchanged global geometry and normal memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .daily_detector import DailyDeepEnergyDetector
from .evidence_experiment import evidence_path
from .experiments import sha256, write_json
from .hierarchical_experiment import minute_groups
from .hierarchical_scan import edge_scores
from .prototype_temporal import frozen_thresholds
from .service_risk_keys import PROTOCOL_COLUMNS, service_keys
from .source_score_scale import SourceScoreScale
from .window_bridge import aggregate_connections
from .window_replay import run as evaluate_windows

DEEP1, DEEP8 = "deep_mahalanobis_m1_ensemble", "deep_mahalanobis_m8_ensemble"
PROFILES = (
    ("deep8_peak_endpoint", DEEP8, "peak_gated", "endpoint", "raw"),
    ("deep8_sum_endpoint", DEEP8, "sum_weak", "endpoint", "raw"),
    ("deep8_sum_service", DEEP8, "sum_weak", "service", "raw"),
    ("deep1_sum_current_service", DEEP1, "sum_weak", "service", "current"),
)


def experiment_profiles(factorial=False):
    if not factorial:
        return PROFILES
    # Complete the initiation/accumulation comparison without changing any
    # source fit, row score, distance, gate or feedback definition.
    extra = tuple(
        (f"deep8_{mode}_{initiation}_{route}", DEEP8, f"{mode}_{initiation}", route, "raw")
        for mode, initiation in (("peak", "weak"), ("sum", "gated"))
        for route in ("endpoint", "service")
    )
    return (*PROFILES, ("deep8_peak_service", DEEP8, "peak_gated", "service", "raw"), *extra)


def parent_name(head, rule, route, guard):
    if guard == "current":
        return f"{head}_current_guard_service_sum_weak_source"
    return f"{head}_sparse_top3_{'service_' if route == 'service' else ''}{rule}_source"


def window_values(values, reference, minutes, groups):
    return np.array(
        [
            aggregate_connections(values[groups[int(minute)]], reference, "top3")
            if int(minute) in groups
            else 0.0
            for minute in minutes
        ]
    )


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.workers)
    profiles = experiment_profiles(getattr(args, "evidence_factorial", False))
    control, output = Path(args.control), Path(args.output)
    registration = json.loads((control / "registration.json").read_text())
    prototype = Path(registration["args"]["prototype"])
    normal_model = torch.load(control / "current_view_detector.pt", weights_only=True)
    risk_root = Path(normal_model["protocol"]["args"]["run"])
    risk_registration = json.loads((risk_root / "registration.json").read_text())
    data_path, key_path = (Path(risk_registration["args"][key]) for key in ("data", "keys"))
    if (
        sha256(data_path) != registration["data_sha256"]
        or sha256(key_path) != registration["keys_sha256"]
    ):
        raise ValueError("Source-score scaling observations or endpoint metadata changed")
    if sha256(risk_root / "row_scores.pt") != registration["row_cache_sha256"]:
        raise ValueError("Source-score scaling original row energies changed")
    cache = torch.load(risk_root / "row_scores.pt", weights_only=True)
    original_distances = torch.load(prototype / "prototype_distances.pt", weights_only=True)
    current_distances = torch.load(control / "current_normal_distances.pt", weights_only=True)
    for stored in (original_distances, current_distances):
        if not torch.equal(cache["combined_indices"], stored["combined_indices"]):
            raise ValueError("Source-score scaling normal views have a different row census")
    combined = cache["combined_indices"].numpy()
    base = normal_model["base_encoder"]
    with np.load(data_path) as payload:
        features, signature, times_all = (
            payload["features"],
            payload["signature"],
            payload["observed_at"],
        )
        names, row_labels = payload["feature_names"].tolist(), payload["label"]
    if tuple(names) != tuple(base["feature_names"]):
        raise ValueError("Source-score scaling feature schema changed")
    reference_rows = base["source_role_indices"]["selection"].numpy()
    fit_positions = np.argsort(signature[reference_rows])[:20000]
    selected = reference_rows[fit_positions]
    if np.any(row_labels[reference_rows] != 0):
        raise ValueError("Source-score profiles contain attack fitting examples")
    audit_ids = np.linspace(0, len(reference_rows) - 1, 128, dtype=np.int64)
    repeated = edge_scores(
        DailyDeepEnergyDetector(base), torch.from_numpy(features[reference_rows[audit_ids]])
    )
    alignment = {}
    for head in (DEEP1, DEEP8):
        expected = cache["row_reference"][head].numpy()
        if len(expected) != len(reference_rows):
            raise ValueError(
                "Source-score profile rows do not align with the original normal reference"
            )
        alignment[head] = float(
            np.max(np.abs(repeated[head] - expected[audit_ids]) / (1 + np.abs(expected[audit_ids])))
        )
        if alignment[head] > 1e-5:
            raise ValueError("Source-score profile grouping does not match actual source NN scores")
    columns = [names.index(k) for k in (*PROTOCOL_COLUMNS, "available_dst_port")]
    fields = features[:, columns]
    del features
    with np.load(key_path) as payload:
        if not np.array_equal(signature, payload["signature"]):
            raise ValueError("Source-score scaling keys have different signatures")
        keys_all = {key: payload[key] for key in ("source", "destination", "port")}
    reference_fields, combined_fields = fields[reference_rows], fields[combined]
    reference_groups = service_keys(
        {k: v[reference_rows] for k, v in keys_all.items()},
        reference_fields[:, :4],
        reference_fields[:, 4],
        "source",
    )
    selected_groups = [reference_groups[i] for i in fit_positions]
    keys = {k: v[combined] for k, v in keys_all.items()}
    row_groups = service_keys(keys, combined_fields[:, :4], combined_fields[:, 4], "source")
    times = times_all[combined]
    groups = minute_groups(times // 60)
    old39 = torch.load(control / "scores.pt", weights_only=True)
    minutes, labels = old39["minutes"].numpy(), old39["labels"].numpy()
    if times_all[selected].max() >= int(minutes.min()) * 60:
        raise ValueError("Source-score fitting entered the target interval")
    output.mkdir(parents=True, exist_ok=False)
    common = {
        "args": vars(args),
        "dataset": "cic",
        "data_sha256": registration["data_sha256"],
        "keys_sha256": registration["keys_sha256"],
        "row_cache_sha256": registration["row_cache_sha256"],
        "source_selection_indices_count": len(selected),
        "original_reference_indices_count": len(reference_rows),
        "fit_selection_mapping": "exact positions within complete saved source-selection order",
        "source_alignment_scaled_errors": alignment,
        "profiles": profiles,
        "score_policies": ["global", "service", "source_service"],
        "minimum_normal_samples": 64,
        "shrinkage_count": 64,
        "aggregation": "top3",
        "fit": "source selection only; unchanged NN/global support/normal geometry",
        "control_policy": (
            "require every available historical match; explicitly list new combinations"
        ),
    }
    write_json(output / "registration.json", common)
    models, scores, references, bases = {}, {}, {}, {}
    for head in (DEEP1, DEEP8):
        values = cache["row_scores"][head].numpy()
        reference = cache["row_reference"][head].numpy()
        parent = old39["methods"][f"{head}_sparse_top3"]
        cal_minutes = parent["source_calibration_minutes"].numpy()
        for policy in ("global", "service", "source_service"):
            name = f"{head}_{policy}"
            model = SourceScoreScale.fit(
                reference[fit_positions],
                selected_groups,
                policy,
                global_q95=float(np.quantile(reference, 0.95)),
            )
            models[name] = model.state_dict()
            scores[name] = model.transform(values, row_groups)
            references[name] = model.transform(reference, reference_groups)
            bases[name] = {
                "energy": torch.from_numpy(
                    window_values(scores[name], references[name], minutes, groups)
                ),
                "observed": parent["observed"].clone(),
                "source_calibration": torch.from_numpy(
                    window_values(scores[name], references[name], cal_minutes, groups)
                ),
                "source_calibration_minutes": parent["source_calibration_minutes"].clone(),
            }
            if policy == "global" and any(
                not torch.equal(bases[name][field], parent[field])
                for field in ("energy", "source_calibration")
            ):
                raise ValueError(
                    "Global identity score policy changed an original top3 window energy"
                )
    torch.save(
        {
            "format_version": 1,
            "normal_model": normal_model,
            "profiles": models,
            "row_references": {k: torch.from_numpy(v) for k, v in references.items()},
            "source_selection_indices": torch.from_numpy(selected),
            "original_reference_indices": torch.from_numpy(reference_rows),
        },
        output / "source_scale_detector.pt",
    )
    restored = torch.load(output / "source_scale_detector.pt", weights_only=True)
    query_ids = np.linspace(0, len(combined) - 1, 128, dtype=np.int64)
    query_groups = [row_groups[i] for i in query_ids]
    for name, state in restored["profiles"].items():
        head = DEEP1 if name.startswith(DEEP1) else DEEP8
        actual = SourceScoreScale(state).transform(
            cache["row_scores"][head].numpy()[query_ids], query_groups
        )
        if not np.array_equal(actual, scores[name][query_ids]):
            raise ValueError("Source-score profile disk restore changed actual held-out transforms")
    for capacity in (39, 50):
        source = control if capacity == 39 else control / "capacity50"
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir()
            write_json(destination / "registration.json", {**common, "capacity": capacity})
        checkpoints = destination / "checkpoints"
        checkpoints.mkdir()
        old = torch.load(source / "scores.pt", weights_only=True)
        if any(not torch.equal(old[field], old39[field]) for field in ("minutes", "labels")):
            raise ValueError("Source-score policies changed target windows between capacities")
        requested_parents = [
            f"{parent_name(head, rule, route, guard)}_{feedback}"
            for identity, head, rule, route, guard in profiles
            for feedback in ("frozen", "candidate")
        ]
        parent_names = [name for name in requested_parents if name in old["methods"]]
        missing_parents = [name for name in requested_parents if name not in old["methods"]]
        required_parents = [
            f"{parent_name(head, rule, route, guard)}_{feedback}"
            for _, head, rule, route, guard in PROFILES
            for feedback in ("frozen", "candidate")
        ]
        if any(name not in old["methods"] for name in required_parents):
            raise ValueError("A required original score-scale control is missing")
        if missing_parents and not getattr(args, "evidence_factorial", False):
            raise ValueError("A required original score-scale control is missing")
        archive = {name: old["methods"][name] for name in ("raw_knn", DEEP1, DEEP8, *parent_names)}
        diagnostics, global_matches = {}, {}
        for identity, head, rule, route, guard in profiles:
            filter_ratio = original_distances["ratios"]["source"]["raw"].numpy()
            guard_ratio = (
                current_distances["ratios"]["source"]["current"].numpy()
                if guard == "current"
                else None
            )
            risk_keys = row_groups if route == "service" else None
            mode, initiation = rule.split("_")
            for policy in ("global", "service", "source_service"):
                key = f"{head}_{policy}"
                base_window = bases[key]
                thresholds = frozen_thresholds(
                    base_window["energy"].numpy(),
                    base_window["observed"].numpy(),
                    labels,
                    base_window["source_calibration"].tolist(),
                    capacity,
                )
                name = f"{head}_normal_units_{policy}_{identity}"
                energy, diagnostics[name] = evidence_path(
                    scores[key],
                    references[key],
                    filter_ratio,
                    minutes,
                    groups,
                    keys,
                    times,
                    thresholds,
                    "source",
                    mode,
                    initiation,
                    checkpoints / f"{name}.pt",
                    aggregation="top3",
                    risk_key_values=risk_keys,
                    guard_ratios=guard_ratio,
                )
                matched = parent_name(head, rule, route, guard)
                if policy == "global" and f"{matched}_frozen" in old["methods"]:
                    expected = old["methods"][f"{matched}_frozen"]["energy"].numpy()
                    if not np.array_equal(energy, expected):
                        raise ValueError(
                            "Global identity score policy changed a matched causal trajectory"
                        )
                    global_matches[name] = matched
                for feedback in ("frozen", "candidate"):
                    archive[f"{name}_{feedback}"] = {
                        **base_window,
                        "energy": torch.from_numpy(energy.copy()),
                        "feedback_energy": base_window["energy"].clone()
                        if feedback == "frozen"
                        else torch.from_numpy(energy.copy()),
                    }
        torch.save({**old, "methods": archive}, destination / "scores.pt")
        evaluate_windows(
            argparse.Namespace(
                run=str(destination),
                output=str(destination / "evaluation"),
                buffer=capacity,
                fpr=0.05,
                history_policy="continuous",
            )
        )
        actual = torch.load(destination / "evaluation/decisions.pt", weights_only=True)
        expected = torch.load(source / "evaluation/decisions.pt", weights_only=True)
        for name in ("raw_knn", DEEP1, DEEP8, *parent_names):
            if any(
                not torch.equal(value, actual[name][field])
                for field, value in expected[name].items()
            ):
                raise ValueError("Source-score comparison changed a preserved parent decision")
        for name, matched in global_matches.items():
            for feedback in ("frozen", "candidate"):
                if any(
                    not torch.equal(value, actual[f"{name}_{feedback}"][field])
                    for field, value in expected[f"{matched}_{feedback}"].items()
                ):
                    raise ValueError("Global identity score policy changed matched decision flags")
        report = json.loads((destination / "evaluation/report.json").read_text())
        write_json(
            destination / "report.json",
            {
                **report,
                "all_global_identity_trajectories_and_decisions_exact": True,
                "global_identity_control_scope": "all available historical controls",
                "historical_controls_not_available_for_new_combinations": missing_parents,
                "all_preserved_parent_decisions_exact": True,
                "actual_source_profile_disk_restore_exact": True,
                "path_diagnostics": diagnostics,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--evidence-factorial", action="store_true")
    run(parser.parse_args())
