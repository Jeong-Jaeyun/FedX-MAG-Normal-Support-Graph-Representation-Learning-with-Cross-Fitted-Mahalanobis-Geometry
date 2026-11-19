"""Chronological alarm-contributor memory around unchanged frozen row energies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .causal_risk import CausalPeakMemory, scan_contributors
from .daily_detector import DailyDeepEnergyDetector
from .experiments import sha256, write_json
from .hierarchical_experiment import minute_groups
from .hierarchical_scan import edge_scores
from .window_bridge import aggregate_connections, replay_windows
from .window_replay import run as evaluate_windows


def replay_risk(
    values,
    reference,
    keys,
    times,
    groups,
    target_minutes,
    source_minutes,
    labels,
    base_flags,
    scope,
    checkpoint,
    *,
    aggregation="rank_scan",
    clear_on_normal=True,
    current_gate_quantile=None,
):
    """Labels reach feedback only after the current energy has been produced."""
    memory = CausalPeakMemory(float(np.quantile(reference, 0.95)))
    if current_gate_quantile is not None and not 0 < current_gate_quantile < 1:
        raise ValueError("Current-evidence gate must be a normal-reference quantile")
    current_gate = (
        float(np.quantile(reference, current_gate_quantile))
        if current_gate_quantile is not None
        else None
    )
    scores, changed = {}, 0
    source_set = set(map(int, source_minutes))
    target_index = {int(t): i for i, t in enumerate(target_minutes)}
    for t in sorted(set(source_set) | set(target_index)):
        positions = groups.get(t, np.empty(0, dtype=np.int64))
        row_values = values[positions]
        source = keys["source"][positions].tolist()
        active = (
            source
            if scope == "source"
            else list(zip(source, keys["destination"][positions].tolist(), strict=True))
        )
        shadow = None
        if t in target_index and target_index[t] == len(target_minutes) // 2:
            torch.save(memory.state_dict(), checkpoint)
            shadow = CausalPeakMemory.from_state_dict(torch.load(checkpoint, weights_only=True))
        augmented = memory.predict((t + 1) * 60, active, row_values, times[positions])
        if shadow is not None:
            resumed = shadow.predict((t + 1) * 60, active, row_values, times[positions])
            if not np.array_equal(augmented, resumed):
                raise ValueError("Risk checkpoint changed the next actual window prediction")
            memory = shadow
        if current_gate is not None:
            augmented[row_values < current_gate] = row_values[row_values < current_gate]
        scores[t] = (
            aggregate_connections(augmented, reference, aggregation) if len(positions) else 0.0
        )
        changed += int(np.any(augmented != row_values))
        # The base gate was independently replayed with strictly prior normal
        # feedback. Augmented alarms never refresh their own state.
        gate = t in target_index and base_flags[target_index[t]]
        contributors = (
            scan_contributors(row_values, reference, aggregation)
            if gate
            else np.empty(0, dtype=np.int64)
        )
        verified_normal = t in source_set or (clear_on_normal and labels[target_index[t]] == 0)
        memory.feedback(verified_normal=verified_normal, contributors=contributors)
    return scores, {
        "changed_windows": changed,
        "maximum_state_keys": memory.maximum_seen,
        "checkpoint_resumed_exact": True,
    }


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.workers)
    data_path, key_path, artifact, control, output = map(
        lambda p: Path(p).resolve(),
        (args.data, args.keys, args.detector, args.control, args.output),
    )
    base = DailyDeepEnergyDetector.load(artifact)
    origin = json.loads((control / "registration.json").read_text())
    data_hash = sha256(data_path)
    if data_hash != base.state["source_data_sha256"] or data_hash != origin["data_sha256"]:
        raise ValueError("Risk comparison changed observations")
    if sha256(artifact) != origin["detector_sha256"] or origin["aggregation"] not in (
        "rank_scan",
        "top3",
    ):
        raise ValueError("Risk comparison needs an original supported aggregation")
    with np.load(data_path) as payload:
        data = {k: payload[k] for k in payload.files}
    with np.load(key_path) as payload:
        keys = {k: payload[k] for k in ("signature", "source", "destination")}
    if not np.array_equal(data["signature"], keys["signature"]):
        raise ValueError("Risk endpoint keys are misaligned")
    if tuple(data["feature_names"]) != base.feature_names:
        raise ValueError("Risk input schema differs")
    stored = torch.load(control / "scores.pt", weights_only=True)
    roles = {k: v.numpy() for k, v in base.state["source_role_indices"].items()}
    target_minutes = stored["minutes"].numpy()
    if any(np.any(data["label"][v] != 0) for v in roles.values()):
        raise ValueError("Risk source roles contain attack rows")
    if len(np.unique(np.concatenate(list(roles.values())))) != sum(map(len, roles.values())):
        raise ValueError("Risk source roles overlap")
    if any(data["observed_at"][v].max() >= target_minutes.min() * 60 for v in roles.values()):
        raise ValueError("Risk fitting entered the target interval")
    minutes = data["observed_at"] // 60
    target = np.flatnonzero(np.isin(minutes, target_minutes))
    combined = np.concatenate((roles["calibration"], target))
    selected = roles["selection"]
    features = torch.from_numpy(data["features"])
    output.mkdir(parents=True, exist_ok=False)
    registration = {
        "args": vars(args),
        "dataset": str(data["dataset"]),
        "data_sha256": data_hash,
        "detector_sha256": sha256(artifact),
        "keys_sha256": sha256(key_path),
        "control_scores_sha256": sha256(control / "scores.pt"),
        "census_sha256": origin["census_sha256"],
        "aggregation": origin["aggregation"],
        "model": "frozen DNN plus deterministic decaying base-alarm contributor memory",
        "risk_parameters": {
            "horizon": 300,
            "decay": 300,
            "maximum_keys": 5000,
            "center": "source selection row-energy q95",
            "scopes": ["source", "peer"],
        },
        "gate": "original flat alarms, separately recomputed for each39/50 capacity",
        "feedback": "prediction then verified whole-normal feedback; clear current keys only",
        "no_self_excitation": (
            "only BASE row energies and actual BASE scan contributors enter memory"
        ),
    }
    write_json(output / "registration.json", registration)
    print("risk frozen edge queries", len(combined), flush=True)
    row_scores = edge_scores(base, features[combined])
    reference = edge_scores(base, features[selected])
    groups = minute_groups(minutes[combined])
    source_minutes = stored["methods"]["raw_knn"]["source_calibration_minutes"].numpy()
    all_groups = minute_groups(minutes)
    membership = np.zeros(len(minutes), dtype=bool)
    membership[roles["calibration"]] = True
    if any(not membership[all_groups[int(t)]].all() for t in source_minutes):
        raise ValueError("Risk calibration minute mixes roles")
    model_state = {
        "format_version": 1,
        "base_encoder": base.state,
        "row_reference": {k: torch.from_numpy(v) for k, v in reference.items()},
        "risk_parameters": registration["risk_parameters"],
    }
    torch.save(model_state, output / "risk_detector.pt")
    torch.save(
        {
            "data_sha256": data_hash,
            "detector_sha256": sha256(artifact),
            "combined_indices": torch.from_numpy(combined.copy()),
            "row_scores": {k: torch.from_numpy(v) for k, v in row_scores.items()},
            "row_reference": model_state["row_reference"],
        },
        output / "row_scores.pt",
    )
    flat_errors = {}
    for name, values in row_scores.items():
        for field, ts in (("energy", target_minutes), ("source_calibration", source_minutes)):
            actual = np.array(
                [
                    aggregate_connections(
                        values[groups[int(t)]], reference[name], origin["aggregation"]
                    )
                    if int(t) in groups
                    else 0.0
                    for t in ts
                ]
            )
            expected = stored["methods"][name][field].numpy()
            error = float(np.max(np.abs(actual - expected) / (1 + np.abs(expected))))
            if error > 1e-8:
                raise ValueError(f"Risk flat energy differs: {name}/{field}/{error}")
            flat_errors[f"{name}/{field}"] = error
    reloaded = torch.load(output / "risk_detector.pt", weights_only=True)
    restored = DailyDeepEnergyDetector(reloaded["base_encoder"])
    ids = np.linspace(0, len(combined) - 1, 128, dtype=np.int64)
    rescored = edge_scores(restored, features[combined[ids]])
    restore_errors = {}
    for name, values in row_scores.items():
        expected = values[ids]
        error = float(np.max(np.abs(rescored[name] - expected) / (1 + np.abs(expected))))
        if error > 1e-5:
            raise ValueError(f"Restored risk encoder differs: {name}/{error}")
        restore_errors[name] = error
    labels = stored["labels"].numpy()
    combined_keys = {k: keys[k][combined] for k in ("source", "destination")}
    times = data["observed_at"][combined]
    for capacity in (39, 50):
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir(exist_ok=False)
            write_json(destination / "registration.json", {**registration, "capacity": capacity})
        checkpoints = destination / "checkpoints"
        checkpoints.mkdir(exist_ok=False)
        archive, matched, diagnostics = {}, {}, {}
        for name, item in stored["methods"].items():
            flags, valid = replay_windows(
                item["energy"].numpy(),
                item["observed"].numpy(),
                labels,
                np.zeros(len(target_minutes), dtype=int),
                item["source_calibration"].tolist(),
                capacity=capacity,
                policy="continuous",
                fpr=0.05,
            )
            archive[name] = {
                **item,
                "flags": torch.from_numpy(flags),
                "valid": torch.from_numpy(valid),
            }
            matched[name] = {"flags": archive[name]["flags"], "valid": archive[name]["valid"]}
        for name, values in row_scores.items():
            for scope in ("source", "peer"):
                head = f"{name}_risk_{scope}"
                scores, diagnostics[head] = replay_risk(
                    values,
                    reloaded["row_reference"][name].numpy(),
                    combined_keys,
                    times,
                    groups,
                    target_minutes,
                    source_minutes,
                    labels,
                    archive[name]["flags"].numpy(),
                    scope,
                    checkpoints / f"{head}.pt",
                    aggregation=origin["aggregation"],
                )
                calibration = torch.tensor(
                    [scores[int(t)] for t in source_minutes], dtype=torch.float64
                )
                if not torch.equal(calibration, stored["methods"][name]["source_calibration"]):
                    raise ValueError("Risk changed wholly normal source calibration energies")
                archive[head] = {
                    "energy": torch.tensor(
                        [scores[int(t)] for t in target_minutes], dtype=torch.float64
                    ),
                    "observed": stored["methods"][name]["observed"].clone(),
                    "source_calibration": calibration,
                    "source_calibration_minutes": torch.from_numpy(source_minutes.copy()),
                }
        torch.save({**stored, "methods": archive}, destination / "scores.pt")
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
        for name, item in matched.items():
            for key, expected in item.items():
                if not torch.equal(decisions[name][key], expected):
                    raise ValueError(f"Risk matched control differs: {name}/{key}")
        report = json.loads((destination / "evaluation/report.json").read_text())
        write_json(
            destination / "report.json",
            {
                **report,
                "capacity": capacity,
                "flat_control_scaled_errors": flat_errors,
                "restore_row_scaled_errors": restore_errors,
                "all_control_decisions_exact_at_matched_buffer": True,
                "risk_diagnostics": diagnostics,
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "keys", "detector", "control", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
