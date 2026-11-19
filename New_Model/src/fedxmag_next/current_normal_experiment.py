"""Separate current-flow normality from unchanged full-context anomaly scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evidence_experiment import evidence_path
from .experiments import sha256, write_json
from .hierarchical_experiment import minute_groups
from .normal_views import CurrentNormalProjector
from .prototype_guard import SCOPES, VerifiedPrototypeMemory, fit_radii
from .prototype_temporal import frozen_thresholds, replay_path
from .service_risk_keys import PROTOCOL_COLUMNS, service_keys
from .sparse_cic_experiment import HEADS
from .window_replay import run as evaluate_windows


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.workers)
    prototype, control, output = map(Path, (args.prototype, args.control, args.output))
    registration = json.loads((prototype / "registration.json").read_text())
    risk_root = Path(registration["args"]["run"])
    risk_registration = json.loads((risk_root / "registration.json").read_text())
    data_path, key_path = (Path(risk_registration["args"][k]) for k in ("data", "keys"))
    if (
        sha256(data_path) != registration["data_sha256"]
        or sha256(key_path) != registration["keys_sha256"]
    ):
        raise ValueError("Current-normal view observations or endpoint keys changed")
    if sha256(risk_root / "row_scores.pt") != registration["row_cache_sha256"]:
        raise ValueError("Current-normal view original anomaly cache changed")
    initial = torch.load(prototype / "prototype_risk_detector.pt", weights_only=True)
    cache = torch.load(risk_root / "row_scores.pt", weights_only=True)
    old_distances = torch.load(prototype / "prototype_distances.pt", weights_only=True)
    if not torch.equal(cache["combined_indices"], old_distances["combined_indices"]):
        raise ValueError("Current-normal view row census differs from parent")
    combined = cache["combined_indices"].numpy()
    original = torch.load(control / "scores.pt", weights_only=True)
    minutes, labels = original["minutes"].numpy(), original["labels"].numpy()
    with np.load(data_path) as payload:
        features = torch.from_numpy(payload["features"])
        times_all, signature = payload["observed_at"], payload["signature"]
        names = payload["feature_names"].tolist()
    if tuple(names) != tuple(initial["base_encoder"]["feature_names"]):
        raise ValueError("Current-normal view feature ordering changed")
    columns = [names.index(k) for k in (*PROTOCOL_COLUMNS, "available_dst_port")]
    fields = features[:, columns].numpy()[combined]
    with np.load(key_path) as payload:
        if not np.array_equal(signature, payload["signature"]):
            raise ValueError("Current-normal view endpoint signatures differ")
        keys = {k: payload[k] for k in ("source", "destination")}
        ports = payload["port"][combined]
    local_keys = {k: v[combined] for k, v in keys.items()}
    routed = {
        scope: service_keys({**local_keys, "port": ports}, fields[:, :4], fields[:, 4], scope)
        for scope in SCOPES
    }
    times = times_all[combined]
    groups = minute_groups(times // 60)
    target = np.flatnonzero(np.isin(times // 60, minutes))
    expected = np.flatnonzero(np.isin(times_all // 60, minutes))
    if not np.array_equal(combined[target], expected):
        raise ValueError("Current-normal view lost or reordered an original target observation")
    mapped = np.full(len(combined), -1, dtype=np.int64)
    mapped[target] = np.arange(len(target))
    reference = initial["source_reference_indices"].numpy()
    selected = initial["base_encoder"]["source_role_indices"]["selection"].numpy()
    if max(times_all[reference].max(), times_all[selected].max()) >= int(minutes.min()) * 60:
        raise ValueError("Current-normal source geometry entered the target interval")

    def subset_keys(ids):
        return {k: v[ids] for k, v in keys.items()}

    output.mkdir(parents=True, exist_ok=False)
    projector = CurrentNormalProjector(initial["base_encoder"])
    print("current-normal source projection", len(reference), len(selected), flush=True)
    normal, selection = (
        projector.project(features[reference]),
        projector.project(features[selected]),
    )
    if not torch.equal(normal["raw"], initial["normal_reference"]["raw"]):
        raise ValueError("Current-normal source normalization changed the original raw seed")
    radii, counts = fit_radii(normal, selection, subset_keys(reference), subset_keys(selected))
    del selection
    radius_errors = {
        scope: abs(radii[scope]["raw"] - initial["radii"][scope]["raw"]) for scope in SCOPES
    }
    if any(error > 1e-10 for error in radius_errors.values()):
        raise ValueError("Current-normal control source radius differs from its original fit")
    print("current-normal source radii", radii, flush=True)
    common = {
        "args": vars(args),
        "dataset": "cic",
        "data_sha256": registration["data_sha256"],
        "keys_sha256": registration["keys_sha256"],
        "row_cache_sha256": registration["row_cache_sha256"],
        "source_detector_sha256": registration["detector_sha256"],
        "current_columns": [names[i] for i in projector.state["columns"]],
        "excluded_context_columns": [name for name in names if name.startswith("context_")],
        "source_raw_radius_errors": radius_errors,
        "source_radius_finite_counts": counts,
        "change": "current-flow marginal normal view only; full-context NN scores fixed",
        "aggregation": "top3",
        "normality_designs": ["current_guard", "current_filter_guard"],
    }
    write_json(output / "registration.json", common)
    model = {
        **initial,
        "radii": radii,
        "normal_reference": normal,
        "current_projection_state": projector.state,
        "protocol": {
            **initial["protocol"],
            "metric": "source1-mode whitened full raw and marginal current-flow normal views",
            "current_view_intervention": common,
        },
    }
    torch.save(model, output / "current_view_detector.pt")
    print("current-normal target projection", len(target), flush=True)
    projected = projector.project(features[combined[target]])
    saved = torch.load(output / "current_view_detector.pt", weights_only=True)
    restored_projector = CurrentNormalProjector(
        saved["base_encoder"], projection_state=saved["current_projection_state"]
    )
    audit_ids = np.linspace(0, len(target) - 1, 128, dtype=np.int64)
    repeated = restored_projector.project(features[combined[target[audit_ids]]])
    projection_errors = {
        k: float(((repeated[k] - v[audit_ids]).abs() / (1 + v[audit_ids].abs())).max())
        for k, v in projected.items()
    }
    if any(error > 1e-10 for error in projection_errors.values()):
        raise ValueError("Current-normal saved projection changed actual held-out queries")
    del features
    memory = VerifiedPrototypeMemory(normal, subset_keys(reference), times_all[reference], radii)
    ratios = {
        scope: {
            view: torch.full((len(combined),), torch.inf, dtype=torch.float64) for view in normal
        }
        for scope in SCOPES
    }
    midpoint_exact = False
    for i, minute in enumerate(minutes):
        ids = groups.get(int(minute), np.empty(0, dtype=np.int64))
        local = mapped[ids]
        if np.any(local < 0):
            raise ValueError("Current-normal memory queried a partially projected target window")
        observations = {k: v[local] for k, v in projected.items()}
        shadow = None
        if i == len(minutes) // 2:
            torch.save(memory.state_dict(), output / "normal_midpoint.pt")
            shadow = VerifiedPrototypeMemory.from_state_dict(
                torch.load(output / "normal_midpoint.pt", weights_only=True)
            )
        result = memory.query(
            (int(minute) + 1) * 60, observations, subset_keys(combined[ids]), times[ids]
        )
        if shadow is not None:
            resumed = shadow.query(
                (int(minute) + 1) * 60, observations, subset_keys(combined[ids]), times[ids]
            )
            if any(not torch.equal(result[s][v], resumed[s][v]) for s in SCOPES for v in normal):
                raise ValueError("Actual current-normal memory disk resume changed next distances")
            memory, midpoint_exact = shadow, True
        for scope in SCOPES:
            for view in normal:
                ratios[scope][view][ids] = result[scope][view]
        memory.feedback(verified_normal=bool(labels[i] == 0))
        if i % 500 == 0:
            print("current-normal windows", i, len(minutes), flush=True)
    del projected
    distance_errors = {}
    for scope in SCOPES:
        actual, expected = ratios[scope]["raw"], old_distances["ratios"][scope]["raw"]
        finite = torch.isfinite(expected)
        if not torch.equal(torch.isfinite(actual), finite):
            raise ValueError("Current-normal control changed which original keys are known")
        distance_errors[scope] = float(
            ((actual[finite] - expected[finite]).abs() / (1 + expected[finite].abs())).max()
        )
        if distance_errors[scope] > 1e-10:
            raise ValueError("Current-normal control normal-distance chronology changed")
    torch.save(
        {"combined_indices": cache["combined_indices"], "ratios": ratios},
        output / "current_normal_distances.pt",
    )
    for capacity in (39, 50):
        source = control if capacity == 39 else control / "capacity50"
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir()
            write_json(destination / "registration.json", {**common, "capacity": capacity})
        old = torch.load(source / "scores.pt", weights_only=True)
        if any(not torch.equal(old[k], original[k]) for k in ("minutes", "labels")):
            raise ValueError("Current-normal capacity comparisons have different target windows")
        archive = dict(old["methods"])
        checkpoints = destination / "checkpoints"
        checkpoints.mkdir()
        diagnostics = {}
        for head in HEADS:
            base = old["methods"][f"{head}_sparse_top3"]
            values, reference_values = (
                cache["row_scores"][head].numpy(),
                cache["row_reference"][head].numpy(),
            )
            thresholds = frozen_thresholds(
                base["energy"].numpy(),
                base["observed"].numpy(),
                labels,
                base["source_calibration"].tolist(),
                capacity,
            )
            for scope in SCOPES:
                current_ratio = ratios[scope]["current"].numpy()
                full_ratio = old_distances["ratios"][scope]["raw"].numpy()
                name = f"{head}_current_filter_prototype_{scope}"
                energy, diagnostics[name] = replay_path(
                    values,
                    reference_values,
                    current_ratio,
                    minutes,
                    groups,
                    local_keys,
                    times,
                    "top3",
                    thresholds,
                    "prototype",
                    scope,
                    checkpoints / f"{name}.pt",
                    continuous_filter=True,
                )
                for feedback in ("frozen", "candidate"):
                    archive[f"{name}_{feedback}"] = {
                        **base,
                        "energy": torch.from_numpy(energy.copy()),
                        "feedback_energy": base["energy"].clone()
                        if feedback == "frozen"
                        else torch.from_numpy(energy.copy()),
                    }
                for route in ("endpoint", "service"):
                    risk_keys = routed[scope] if route == "service" else None
                    for rule in ("peak_gated", "sum_weak"):
                        mode, initiation = rule.split("_")
                        tag = "service_" if route == "service" else ""
                        parent_name = f"{head}_sparse_top3_{tag}{rule}_{scope}_frozen"
                        matched, _ = evidence_path(
                            values,
                            reference_values,
                            full_ratio,
                            minutes,
                            groups,
                            local_keys,
                            times,
                            thresholds,
                            scope,
                            mode,
                            initiation,
                            checkpoints / f"matched_{route}_{rule}_{scope}_{head}.pt",
                            aggregation="top3",
                            risk_key_values=risk_keys,
                        )
                        if not np.array_equal(
                            matched, old["methods"][parent_name]["energy"].numpy()
                        ):
                            raise ValueError(
                                "Current-normal full-view matched evidence trajectory changed"
                            )
                        for design, filter_ratio in (
                            ("current_guard", full_ratio),
                            ("current_filter_guard", current_ratio),
                        ):
                            name = f"{head}_{design}_{route}_{rule}_{scope}"
                            energy, diagnostics[name] = evidence_path(
                                values,
                                reference_values,
                                filter_ratio,
                                minutes,
                                groups,
                                local_keys,
                                times,
                                thresholds,
                                scope,
                                mode,
                                initiation,
                                checkpoints / f"{name}.pt",
                                aggregation="top3",
                                risk_key_values=risk_keys,
                                guard_ratios=current_ratio,
                            )
                            for feedback in ("frozen", "candidate"):
                                archive[f"{name}_{feedback}"] = {
                                    **base,
                                    "energy": torch.from_numpy(energy.copy()),
                                    "feedback_energy": base["energy"].clone()
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
        for name, item in expected.items():
            if any(not torch.equal(value, actual[name][key]) for key, value in item.items()):
                raise ValueError("Current-normal view changed a preserved full-view decision")
        report = json.loads((destination / "evaluation/report.json").read_text())
        write_json(
            destination / "report.json",
            {
                **report,
                "all_full_view_parent_trajectories_and_decisions_exact": True,
                "raw_parent_ratio_scaled_errors": distance_errors,
                "saved_projection_scaled_errors": projection_errors,
                "actual_normal_midpoint_resume_exact": midpoint_exact,
                "path_diagnostics": diagnostics,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prototype", "control", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())
