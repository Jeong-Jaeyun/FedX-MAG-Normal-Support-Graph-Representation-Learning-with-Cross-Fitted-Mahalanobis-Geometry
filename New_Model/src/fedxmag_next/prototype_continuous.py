"""Reuse the exact causal prototype cache; remove the local-energy cutoff only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .experiments import sha256, write_json
from .hierarchical_experiment import minute_groups
from .prototype_guard import SCOPES
from .prototype_temporal import frozen_thresholds, replay_path
from .window_replay import run as evaluate_windows


def run(args):
    torch.set_num_threads(args.workers)
    origin, output = Path(args.run).resolve(), Path(args.output).resolve()
    registration = json.loads((origin / "registration.json").read_text())
    risk_root = Path(registration["args"]["run"]).resolve()
    risk_registration = json.loads((risk_root / "registration.json").read_text())
    if sha256(risk_root / "row_scores.pt") != registration["row_cache_sha256"]:
        raise ValueError("Continuous prototype row-energy cache changed")
    data_path = Path(risk_registration["args"]["data"]).resolve()
    key_path = Path(risk_registration["args"]["keys"]).resolve()
    if (
        sha256(data_path) != registration["data_sha256"]
        or sha256(key_path) != registration["keys_sha256"]
    ):
        raise ValueError("Continuous prototype observations changed")
    cache = torch.load(risk_root / "row_scores.pt", weights_only=True)
    distances = torch.load(origin / "prototype_distances.pt", weights_only=True)
    if not torch.equal(cache["combined_indices"], distances["combined_indices"]):
        raise ValueError("Continuous prototype distances cover other row indices")
    combined = cache["combined_indices"].numpy()
    with np.load(data_path) as payload:
        times, signature = payload["observed_at"][combined], payload["signature"]
    with np.load(key_path) as payload:
        if not np.array_equal(signature, payload["signature"]):
            raise ValueError("Continuous prototype keys are misaligned")
        keys = {k: payload[k][combined] for k in ("source", "destination")}
    groups = minute_groups(times // 60)
    output.mkdir(parents=True, exist_ok=False)
    common = {
        "args": vars(args),
        "dataset": registration["dataset"],
        "parent_model_sha256": sha256(origin / "prototype_risk_detector.pt"),
        "parent_distances_sha256": sha256(origin / "prototype_distances.pt"),
        "row_cache_sha256": registration["row_cache_sha256"],
        "change": "continuous local prototype energy for every finite same-key distance",
        "unchanged": "radii,normaladmission,gating,weights,observations,feedback39/50",
        "calibration_policy": registration["calibration_policy"],
    }
    write_json(output / "registration.json", common)
    torch.save(
        {
            "format_version": 1,
            "parent_model": str(origin / "prototype_risk_detector.pt"),
            "parent_model_sha256": common["parent_model_sha256"],
            "continuous_filter": True,
            "protocol": common,
        },
        output / "continuous_prototype_detector.pt",
    )
    for capacity in (39, 50):
        source = origin if capacity == 39 else origin / "capacity50"
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir(exist_ok=False)
            write_json(destination / "registration.json", {**common, "capacity": capacity})
        old = torch.load(source / "scores.pt", weights_only=True)
        minutes, labels = old["minutes"].numpy(), old["labels"].numpy()
        archive, diagnostics, predicted_flags = dict(old["methods"]), {}, {}
        checkpoint_folder = destination / "checkpoints"
        checkpoint_folder.mkdir()
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
                ratio = distances["ratios"][scope][
                    "deep" if name.startswith("deep_") else "raw"
                ].numpy()
                for path in ("prototype", "prototype_risk_fixed"):
                    head = f"{name}_continuous_{path}_{scope}"
                    energy, diagnostics[head] = replay_path(
                        tensor.numpy(),
                        cache["row_reference"][name].numpy(),
                        ratio,
                        minutes,
                        groups,
                        keys,
                        times,
                        registration["aggregation"],
                        thresholds,
                        path,
                        scope,
                        checkpoint_folder / f"{head}.pt",
                        continuous_filter=True,
                    )
                    predicted_flags[head] = (energy > thresholds) & baseline["observed"].numpy()
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
        actual = torch.load(destination / "evaluation/decisions.pt", weights_only=True)
        expected = torch.load(source / "evaluation/decisions.pt", weights_only=True)
        for name, item in expected.items():
            if any(not torch.equal(actual[name][k], v) for k, v in item.items()):
                raise ValueError(f"Continuous prototype parent decision differs: {name}")
        for name, flags in predicted_flags.items():
            if not np.array_equal(actual[name]["flags"].numpy(), flags):
                raise ValueError("Continuous prototype fixed-feedback replay differs")
        for name in cache["row_scores"]:
            for scope in SCOPES:
                p = actual[f"{name}_continuous_prototype_{scope}"]["flags"]
                joint = actual[f"{name}_continuous_prototype_risk_fixed_{scope}"]["flags"]
                if torch.any(p & ~actual[name]["flags"]) or torch.any(p & ~joint):
                    raise ValueError("Continuous normal-filter alarm containment failed")
        report = json.loads((destination / "evaluation/report.json").read_text())
        write_json(
            destination / "report.json",
            {
                **report,
                "all_parent_decisions_exact": True,
                "normal_filter_alarm_subset_exact": True,
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
