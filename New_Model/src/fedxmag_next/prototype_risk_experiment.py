"""Measured past-normal prototypes, filtering, and bounded temporal evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .daily_detector import DailyDeepEnergyDetector
from .experiments import sha256, write_json
from .hierarchical_experiment import minute_groups
from .hierarchical_scan import edge_scores
from .prototype_guard import SCOPES, PrototypeProjector, VerifiedPrototypeMemory, fit_radii
from .prototype_temporal import PATHS, frozen_thresholds, replay_path
from .window_replay import run as evaluate_windows


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.workers)
    origin, output = Path(args.run).resolve(), Path(args.output).resolve()
    registration = json.loads((origin / "registration.json").read_text())
    data_path = Path(registration["args"]["data"]).resolve()
    key_path = Path(registration["args"]["keys"]).resolve()
    source_path = Path(registration["args"]["detector"]).resolve()
    cache = torch.load(origin / "row_scores.pt", weights_only=True)
    if sha256(data_path) != cache["data_sha256"] or sha256(key_path) != registration["keys_sha256"]:
        raise ValueError("Prototype experiment observation metadata changed")
    if sha256(source_path) != cache["detector_sha256"]:
        raise ValueError("Prototype experiment source model changed")
    base = torch.load(source_path, weights_only=True)
    with np.load(data_path) as payload:
        data = {
            k: payload[k]
            for k in ("features", "signature", "observed_at", "label", "feature_names")
        }
    with np.load(key_path) as payload:
        keys = {k: payload[k] for k in ("signature", "source", "destination")}
    if not np.array_equal(keys["signature"], data["signature"]):
        raise ValueError("Prototype experiment endpoint keys are misaligned")
    if tuple(data["feature_names"]) != tuple(base["feature_names"]):
        raise ValueError("Prototype experiment feature schema differs")
    stored = torch.load(origin / "scores.pt", weights_only=True)
    minutes = stored["minutes"].numpy()
    roles = {k: v.numpy() for k, v in base["source_role_indices"].items()}
    if any(np.any(data["label"][v] != 0) for v in roles.values()):
        raise ValueError("Prototype source roles include attack examples")
    if any(data["observed_at"][v].max() >= minutes.min() * 60 for v in roles.values()):
        raise ValueError("Prototype source roles entered the target interval")
    if len(np.unique(np.concatenate(list(roles.values())))) != sum(map(len, roles.values())):
        raise ValueError("Prototype source roles overlap")
    train = roles["train"]
    train = train[np.argsort(data["signature"][train])[:20000]]
    reference = train[:5000]
    combined = cache["combined_indices"].numpy()
    combined_times = data["observed_at"][combined]
    groups = minute_groups(combined_times // 60)
    target_positions = np.flatnonzero(np.isin(combined_times // 60, minutes))
    expected_target = np.flatnonzero(np.isin(data["observed_at"] // 60, minutes))
    if not np.array_equal(combined[target_positions], expected_target):
        raise ValueError("Prototype cache lost or reordered target observations")
    mapped = np.full(len(combined), -1, dtype=np.int64)
    mapped[target_positions] = np.arange(len(target_positions))
    features = torch.from_numpy(data["features"])
    source_detector = DailyDeepEnergyDetector(base)
    if not torch.equal(
        source_detector.normalize_features(features[reference]),
        base["raw_reference"][: len(reference)],
    ):
        raise ValueError("Prototype initial rows differ from the actual source reference")
    output.mkdir(parents=True, exist_ok=False)
    common = {
        "args": vars(args),
        "dataset": registration["dataset"],
        "data_sha256": cache["data_sha256"],
        "detector_sha256": cache["detector_sha256"],
        "keys_sha256": registration["keys_sha256"],
        "row_cache_sha256": sha256(origin / "row_scores.pt"),
        "aggregation": registration.get("aggregation", "rank_scan"),
        "calibration_policy": (
            "every candidate uses original frozen energies for past-normal feedback"
        ),
        "normal_memory": "source train5000 seed; FIFO5000; whole-normal admission AFTER prediction",
        "metric": "source1-mode whitened raw / concatenated three whitened64-D DNN states",
        "radius": "same-key source selection nearest-distance q95; unknown identity remains novel",
        "identity": "same-network source/peer equality only; no numeric IP neural coordinate",
        "risk": (
            "TTL300/decay300/5000keys; actual base-alarm contributors; "
            "no normal-clear or self-excitation"
        ),
        "factorial_paths": list(PATHS),
        "scopes": list(SCOPES),
    }
    write_json(output / "registration.json", common)
    projector = PrototypeProjector(base)

    def subset_keys(ids):
        return {k: keys[k][ids] for k in ("source", "destination")}

    print("prototype source projection", len(reference), len(roles["selection"]), flush=True)
    normal = projector.project(features[reference])
    selected = projector.project(features[roles["selection"]])
    radii, counts = fit_radii(
        normal, selected, subset_keys(reference), subset_keys(roles["selection"])
    )
    del selected
    print("source normal radii", radii, "finite counts", counts, flush=True)
    model = {
        "format_version": 1,
        "base_encoder": base,
        "radii": radii,
        "source_reference_indices": torch.from_numpy(reference.copy()),
        "normal_reference": normal,
        "normal_reference_keys": {
            k: torch.from_numpy(v.view(np.int64).copy()) for k, v in subset_keys(reference).items()
        },
        "normal_reference_times": torch.from_numpy(data["observed_at"][reference].copy()),
        "maximum_prototypes": 5000,
        "protocol": common,
    }
    torch.save(model, output / "prototype_risk_detector.pt")
    print("prototype target projection", len(target_positions), flush=True)
    projected = projector.project(features[combined[target_positions]])
    reloaded = torch.load(output / "prototype_risk_detector.pt", weights_only=True)
    restored_projector = PrototypeProjector(reloaded["base_encoder"])
    audit_ids = np.linspace(0, len(target_positions) - 1, 128, dtype=np.int64)
    repeated = restored_projector.project(features[combined[target_positions[audit_ids]]])
    projection_errors = {}
    for name, values in projected.items():
        expected = values[audit_ids]
        error = float(((repeated[name] - expected).abs() / (1 + expected.abs())).max())
        if error > 1e-5:
            raise ValueError(f"Restored prototype projection differs: {name}/{error}")
        projection_errors[name] = error
    original = edge_scores(
        DailyDeepEnergyDetector(reloaded["base_encoder"]),
        features[combined[target_positions[audit_ids]]],
    )
    energy_errors = {}
    for name, values in original.items():
        expected = cache["row_scores"][name][target_positions[audit_ids]].numpy()
        error = float(np.max(np.abs(values - expected) / (1 + np.abs(expected))))
        if error > 1e-5:
            raise ValueError(f"Prototype original cached row energy differs: {name}/{error}")
        energy_errors[name] = error
    memory = VerifiedPrototypeMemory(
        normal, subset_keys(reference), data["observed_at"][reference], radii
    )
    ratios = {
        scope: {
            name: torch.full((len(combined),), torch.inf, dtype=torch.float64) for name in normal
        }
        for scope in SCOPES
    }
    labels = stored["labels"].numpy()
    memory_audit = []
    for i, minute in enumerate(minutes):
        ids = groups.get(int(minute), np.empty(0, dtype=np.int64))
        local = mapped[ids]
        if np.any(local < 0):
            raise ValueError("A prototype target window was only partially projected")
        observations = {k: v[local] for k, v in projected.items()}
        shadow = None
        if i == len(minutes) // 2:
            torch.save(memory.state_dict(), output / "normal_memory_midpoint.pt")
            shadow = VerifiedPrototypeMemory.from_state_dict(
                torch.load(output / "normal_memory_midpoint.pt", weights_only=True)
            )
        result = memory.query(
            (int(minute) + 1) * 60, observations, subset_keys(combined[ids]), combined_times[ids]
        )
        if shadow is not None:
            resumed = shadow.query(
                (int(minute) + 1) * 60,
                observations,
                subset_keys(combined[ids]),
                combined_times[ids],
            )
            if any(not torch.equal(result[s][k], resumed[s][k]) for s in SCOPES for k in normal):
                raise ValueError("Actual prototype memory disk resume differs")
            memory = shadow
        for scope in SCOPES:
            for name in normal:
                ratios[scope][name][ids] = result[scope][name]
        memory_audit.append((int(minute), len(memory.times)))
        memory.feedback(verified_normal=bool(labels[i] == 0))
        if i % 500 == 0:
            print("prototype windows", i, len(minutes), "memory", len(memory.times), flush=True)
    torch.save(
        {
            "combined_indices": cache["combined_indices"],
            "ratios": ratios,
            "memory_audit": torch.tensor(memory_audit),
        },
        output / "prototype_distances.pt",
    )
    del projected
    for capacity in (39, 50):
        origin_capacity = origin if capacity == 39 else origin / "capacity50"
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir(exist_ok=False)
            write_json(destination / "registration.json", {**common, "capacity": capacity})
        checkpoint_folder = destination / "checkpoints"
        checkpoint_folder.mkdir()
        old = torch.load(origin_capacity / "scores.pt", weights_only=True)
        archive, diagnostics = dict(old["methods"]), {}
        expected_flags = {}
        for name, tensor in cache["row_scores"].items():
            baseline = old["methods"][name]
            thresholds = frozen_thresholds(
                baseline["energy"].numpy(),
                baseline["observed"].numpy(),
                labels,
                baseline["source_calibration"].tolist(),
                capacity,
            )
            for scope in SCOPES:
                ratio = ratios[scope]["deep" if name.startswith("deep_") else "raw"].numpy()
                for path in PATHS:
                    head = f"{name}_{path}_{scope}"
                    energy, diagnostics[head] = replay_path(
                        tensor.numpy(),
                        cache["row_reference"][name].numpy(),
                        ratio,
                        minutes,
                        groups,
                        subset_keys(combined),
                        combined_times,
                        common["aggregation"],
                        thresholds,
                        path,
                        scope,
                        checkpoint_folder / f"{head}.pt",
                    )
                    valid = baseline["observed"].numpy() & np.isfinite(thresholds)
                    expected_flags[head] = (energy > thresholds) & valid
                    archive[head] = {
                        "energy": torch.from_numpy(energy),
                        "observed": baseline["observed"].clone(),
                        "source_calibration": baseline["source_calibration"].clone(),
                        "source_calibration_minutes": baseline[
                            "source_calibration_minutes"
                        ].clone(),
                        "feedback_energy": baseline["energy"].clone(),
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
        decisions = torch.load(destination / "evaluation/decisions.pt", weights_only=True)
        expected = torch.load(origin_capacity / "evaluation/decisions.pt", weights_only=True)
        for name, item in expected.items():
            if any(not torch.equal(decisions[name][k], v) for k, v in item.items()):
                raise ValueError(f"Prototype original control changed: {name}")
        for head, flags in expected_flags.items():
            if not np.array_equal(decisions[head]["flags"].numpy(), flags):
                raise ValueError("Saved prototype feedback policy did not replay exactly")
        for name in cache["row_scores"]:
            base_flags = decisions[name]["flags"]
            for scope in SCOPES:
                filtered = decisions[f"{name}_prototype_{scope}"]["flags"]
                if torch.any(filtered & ~base_flags):
                    raise ValueError("Prototype-only filtering created an alarm")
                for path in ("risk_fixed", "guard_risk_fixed"):
                    if torch.any(base_flags & ~decisions[f"{name}_{path}_{scope}"]["flags"]):
                        raise ValueError(
                            "Unfiltered risk lost a frozen alarm under shared thresholds"
                        )
                if torch.any(
                    filtered & ~decisions[f"{name}_prototype_risk_fixed_{scope}"]["flags"]
                ):
                    raise ValueError("Combined risk lost a prototype-only alarm")
        report = json.loads((destination / "evaluation/report.json").read_text())
        write_json(
            destination / "report.json",
            {
                **report,
                "all_original_decisions_exact": True,
                "normal_filter_alarm_subset_exact": True,
                "unfiltered_risk_alarm_superset_exact": True,
                "prototype_memory_maximum": memory.maximum_seen,
                "prototype_midpoint_resume_exact": True,
                "projection_restore_scaled_errors": projection_errors,
                "original_row_energy_restore_scaled_errors": energy_errors,
                "radii": radii,
                "radius_finite_selection_counts": counts,
                "path_diagnostics": diagnostics,
                "calibration_policy": common["calibration_policy"],
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())
