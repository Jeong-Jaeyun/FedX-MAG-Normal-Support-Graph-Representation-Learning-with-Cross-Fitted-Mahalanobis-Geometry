"""Remove normal-confirmation clearing from risk state, retaining calibration protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .experiments import sha256, write_json
from .hierarchical_experiment import minute_groups
from .risk_experiment import replay_risk
from .window_replay import run as evaluate_windows


def run(args):
    root, output = Path(args.run).resolve(), Path(args.output).resolve()
    registration = json.loads((root / "registration.json").read_text())
    cache = torch.load(root / "row_scores.pt", weights_only=True)
    data_path, key_path = map(Path, (registration["args"]["data"], registration["args"]["keys"]))
    if sha256(data_path) != cache["data_sha256"] or sha256(key_path) != registration["keys_sha256"]:
        raise ValueError("Risk sensitivity metadata changed")
    with np.load(data_path) as payload:
        times, signature = payload["observed_at"], payload["signature"]
    with np.load(key_path) as payload:
        if not np.array_equal(signature, payload["signature"]):
            raise ValueError("Sensitivity endpoint keys are misaligned")
        keys = {k: payload[k][cache["combined_indices"].numpy()] for k in ("source", "destination")}
    times = times[cache["combined_indices"].numpy()]
    groups = minute_groups(times // 60)
    output.mkdir(parents=True, exist_ok=False)
    for capacity in (39, 50):
        origin = root if capacity == 39 else root / "capacity50"
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir(exist_ok=False)
        stored = torch.load(origin / "scores.pt", weights_only=True)
        target_minutes = stored["minutes"].numpy()
        labels = stored["labels"].numpy()
        source_minutes = stored["methods"]["raw_knn"]["source_calibration_minutes"].numpy()
        write_json(
            destination / "registration.json",
            {
                "args": vars(args),
                "dataset": registration["dataset"],
                "capacity": capacity,
                "risk_run_scores_sha256": sha256(origin / "scores.pt"),
                "row_cache_sha256": sha256(root / "row_scores.pt"),
                "change": (
                    "risk NEVER clears on target normal verification; actual base gates unchanged"
                ),
                "remaining_oracle": (
                    "original base/window calibration still receives verified normal feedback"
                ),
                "state_budget": "unchanged q95 center / TTL300 / decay300 /5000 keys",
                "current_gate_quantile": args.current_gate_quantile,
            },
        )
        checkpoints = destination / "checkpoints"
        checkpoints.mkdir()
        archive, diagnostics = dict(stored["methods"]), {}
        for name, tensor in cache["row_scores"].items():
            for scope in ("source", "peer"):
                gate_name = (
                    ""
                    if args.current_gate_quantile is None
                    else f"_q{round(100 * args.current_gate_quantile)}"
                )
                head = f"{name}_risk_no_clear{gate_name}_{scope}"
                values, diagnostics[head] = replay_risk(
                    tensor.numpy(),
                    cache["row_reference"][name].numpy(),
                    keys,
                    times,
                    groups,
                    target_minutes,
                    source_minutes,
                    labels,
                    archive[name]["flags"].numpy(),
                    scope,
                    checkpoints / f"{head}.pt",
                    aggregation=registration.get("aggregation", "rank_scan"),
                    clear_on_normal=False,
                    current_gate_quantile=args.current_gate_quantile,
                )
                calibration = torch.tensor(
                    [values[int(t)] for t in source_minutes], dtype=torch.float64
                )
                if not torch.equal(calibration, archive[name]["source_calibration"]):
                    raise ValueError("No-clear risk changed source calibration")
                archive[head] = {
                    "energy": torch.tensor(
                        [values[int(t)] for t in target_minutes], dtype=torch.float64
                    ),
                    "observed": archive[name]["observed"].clone(),
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
        actual = torch.load(destination / "evaluation/decisions.pt", weights_only=True)
        expected = torch.load(origin / "evaluation/decisions.pt", weights_only=True)
        for name, item in expected.items():
            if any(not torch.equal(actual[name][k], v) for k, v in item.items()):
                raise ValueError(f"Sensitivity original decision changed: {name}")
        report = json.loads((destination / "evaluation/report.json").read_text())
        write_json(
            destination / "report.json",
            {
                **report,
                "all_original_decisions_exact": True,
                "risk_diagnostics": diagnostics,
                "remaining_oracle": "verified-normal calibration retained",
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--current-gate-quantile", type=float, default=None)
    run(parser.parse_args())
