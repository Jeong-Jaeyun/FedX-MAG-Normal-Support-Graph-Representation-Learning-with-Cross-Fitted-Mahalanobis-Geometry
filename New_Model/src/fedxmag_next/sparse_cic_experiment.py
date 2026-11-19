"""Test fixed sparse pooling with causal prototypes and original weak evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evidence_experiment import evidence_path
from .experiments import sha256, write_json
from .hierarchical_experiment import minute_groups
from .prototype_guard import SCOPES
from .prototype_temporal import frozen_thresholds, replay_path
from .window_bridge import aggregate_connections
from .window_replay import run as evaluate_windows

HEADS = ("deep_mahalanobis_m1_ensemble", "deep_mahalanobis_m8_ensemble")


def run(args):
    torch.set_num_threads(args.workers)
    prototype, output = Path(args.prototype), Path(args.output)
    registration = json.loads((prototype / "registration.json").read_text())
    risk_root = Path(registration["args"]["run"])
    parent_registration = json.loads((risk_root / "registration.json").read_text())
    if registration["dataset"] != "cic":
        raise ValueError("This fixed pooling hypothesis is a CIC-only comparison")
    data_path, key_path = (Path(parent_registration["args"][k]) for k in ("data", "keys"))
    if (
        sha256(data_path) != registration["data_sha256"]
        or sha256(key_path) != registration["keys_sha256"]
    ):
        raise ValueError("Sparse pooling input observations or keys changed")
    if sha256(risk_root / "row_scores.pt") != registration["row_cache_sha256"]:
        raise ValueError("Sparse pooling row-energy cache changed")
    cache = torch.load(risk_root / "row_scores.pt", weights_only=True)
    distances = torch.load(prototype / "prototype_distances.pt", weights_only=True)
    if not torch.equal(cache["combined_indices"], distances["combined_indices"]):
        raise ValueError("Sparse pooling normal distances have a different row census")
    combined = cache["combined_indices"].numpy()
    with np.load(data_path) as payload:
        times, signature = payload["observed_at"][combined], payload["signature"]
    with np.load(key_path) as payload:
        if not np.array_equal(signature, payload["signature"]):
            raise ValueError("Sparse pooling endpoint keys are misaligned")
        keys = {k: payload[k][combined] for k in ("source", "destination")}
    groups = minute_groups(times // 60)
    old = torch.load(risk_root / "scores.pt", weights_only=True)
    minutes, labels = old["minutes"].numpy(), old["labels"].numpy()
    sparse, errors = {}, {}
    for head in HEADS:
        parent = old["methods"][head]
        values = cache["row_scores"][head].numpy()
        reference = np.sort(cache["row_reference"][head].numpy())
        cal_minutes = parent["source_calibration_minutes"].numpy()

        def windows(queries, aggregation, values=values, reference=reference):
            return np.array(
                [
                    aggregate_connections(
                        values[groups[int(minute)]], reference, aggregation, reference_sorted=True
                    )
                    if int(minute) in groups
                    else 0.0
                    for minute in queries
                ]
            )

        errors[head] = {}
        for field, query_minutes in (("energy", minutes), ("source_calibration", cal_minutes)):
            actual = windows(query_minutes, registration["aggregation"])
            expected = parent[field].numpy()
            error = float(np.max(np.abs(actual - expected) / (1 + np.abs(expected))))
            if error > 1e-5:
                raise ValueError(
                    f"Sparse pooling old aggregation reconstruction failed: {head}/{field}"
                )
            errors[head][field] = error
        sparse[head] = {
            **parent,
            "energy": torch.from_numpy(windows(minutes, "top3")),
            "source_calibration": torch.from_numpy(windows(cal_minutes, "top3")),
        }
    output.mkdir(parents=True, exist_ok=False)
    common = {
        "args": vars(args),
        "dataset": "cic",
        "data_sha256": registration["data_sha256"],
        "row_cache_sha256": registration["row_cache_sha256"],
        "aggregation": "top3",
        "normal_view": "raw",
        "original_aggregation_reconstruction_scaled_errors": errors,
        "change": (
            "only pooling and corresponding source calibration/base gate; "
            "fixed row energies and normal distances"
        ),
    }
    write_json(output / "registration.json", common)
    for capacity in (39, 50):
        source = risk_root if capacity == 39 else risk_root / "capacity50"
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir()
            write_json(destination / "registration.json", {**common, "capacity": capacity})
        checkpoints = destination / "checkpoints"
        checkpoints.mkdir()
        parent = torch.load(source / "scores.pt", weights_only=True)
        archive = {name: parent["methods"][name] for name in ("raw_knn", *HEADS)}
        diagnostics = {}
        for head, base in sparse.items():
            values, reference = (
                cache["row_scores"][head].numpy(),
                cache["row_reference"][head].numpy(),
            )
            archive[f"{head}_sparse_top3"] = base
            thresholds = frozen_thresholds(
                base["energy"].numpy(),
                base["observed"].numpy(),
                labels,
                base["source_calibration"].tolist(),
                capacity,
            )
            for scope in SCOPES:
                ratio = distances["ratios"][scope]["raw"].numpy()
                for rule in ("prototype", "peak_gated", "peak_weak", "sum_weak"):
                    name = f"{head}_sparse_top3_{rule}_{scope}"
                    if rule == "prototype":
                        energy, diagnostics[name] = replay_path(
                            values,
                            reference,
                            ratio,
                            minutes,
                            groups,
                            keys,
                            times,
                            "top3",
                            thresholds,
                            "prototype",
                            scope,
                            checkpoints / f"{name}.pt",
                            continuous_filter=True,
                        )
                    else:
                        mode, initiation = rule.split("_")
                        energy, diagnostics[name] = evidence_path(
                            values,
                            reference,
                            ratio,
                            minutes,
                            groups,
                            keys,
                            times,
                            thresholds,
                            scope,
                            mode,
                            initiation,
                            checkpoints / f"{name}.pt",
                            aggregation="top3",
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
        expected = torch.load(source / "evaluation/decisions.pt", weights_only=True)
        actual = torch.load(destination / "evaluation/decisions.pt", weights_only=True)
        for name in ("raw_knn", *HEADS):
            if any(
                not torch.equal(value, actual[name][key]) for key, value in expected[name].items()
            ):
                raise ValueError("Sparse pooling changed a preserved parent decision")
        report = json.loads((destination / "evaluation/report.json").read_text())
        write_json(
            destination / "report.json",
            {
                **report,
                "all_preserved_parent_decisions_exact": True,
                "original_aggregation_reconstruction_scaled_errors": errors,
                "path_diagnostics": diagnostics,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())
