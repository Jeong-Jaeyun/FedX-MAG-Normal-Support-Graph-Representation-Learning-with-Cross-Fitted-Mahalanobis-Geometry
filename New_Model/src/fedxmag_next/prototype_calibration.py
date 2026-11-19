"""Keep temporal scores fixed and ablate only the final normal-feedback reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .experiments import sha256, write_json
from .window_replay import run as evaluate_windows


def run(args):
    origin, output = Path(args.run).resolve(), Path(args.output).resolve()
    registration = json.loads((origin / "registration.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    for capacity in (39, 50):
        source = origin if capacity == 39 else origin / "capacity50"
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir(exist_ok=False)
        old = torch.load(source / "scores.pt", weights_only=True)
        write_json(
            destination / "registration.json",
            {
                "args": vars(args),
                "dataset": registration["dataset"],
                "capacity": capacity,
                "parent_scores_sha256": sha256(source / "scores.pt"),
                "change": "only final calibration uses prior verified-normal CANDIDATE energies",
                "unchanged": (
                    "all neural/prototype/temporal states, base gates, row and window energies"
                ),
                "source_calibration": "original source energies retained; no warmup removal",
            },
        )
        methods, aliases = dict(old["methods"]), {}
        for name, item in old["methods"].items():
            if "_continuous_prototype" not in name:
                continue
            new_name = name.replace("_continuous_", "_adaptive_continuous_")
            methods[new_name] = {**item, "feedback_energy": item["energy"].clone()}
            aliases[new_name] = name
        if len(aliases) != 16:
            raise ValueError(
                "Calibration ablation requires the prespecified sixteen continuous heads"
            )
        torch.save({**old, "methods": methods}, destination / "scores.pt")
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
                raise ValueError(f"Final calibration changed a parent decision: {name}")
        for name, previous in aliases.items():
            for field in ("energy", "observed", "source_calibration", "source_calibration_minutes"):
                if not torch.equal(methods[name][field], old["methods"][previous][field]):
                    raise ValueError("Final calibration changed a saved observation energy")
        report = json.loads((destination / "evaluation/report.json").read_text())
        write_json(
            destination / "report.json",
            {
                **report,
                "all_parent_decisions_exact": True,
                "all_candidate_energies_exact": True,
                "candidate_to_parent": aliases,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
