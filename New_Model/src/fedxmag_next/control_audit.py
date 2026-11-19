"""Audit exact raw window controls, explicitly matching feedback capacity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .experiments import sha256, write_json
from .window_bridge import replay_windows


def audit(control, candidate, buffer):
    control, candidate = Path(control).resolve(), Path(candidate).resolve()
    old = torch.load(control / "scores.pt", weights_only=True)
    new = torch.load(candidate / "scores.pt", weights_only=True)
    for key in ("minutes", "labels"):
        if not torch.equal(old[key], new[key]):
            raise ValueError(f"Control windows/labels differ: {key}")
    if old["types"] != new["types"]:
        raise ValueError("Control whole-window attack types differ")
    errors = {}
    for name in (
        "raw_knn",
        "raw_mahalanobis_m1",
        "raw_mahalanobis_m8",
        "record_count",
        "sampled_record_count",
    ):
        a, b = old["methods"][name], new["methods"][name]
        errors[name] = {}
        for key in ("energy", "source_calibration"):
            if a[key].shape != b[key].shape:
                raise ValueError(f"Raw control shape differs: {name}/{key}")
            error = float(((a[key] - b[key]).abs() / (1 + a[key].abs())).max())
            if error > 1e-8:
                raise ValueError(f"Raw control energy differs: {name}/{key}/{error}")
            errors[name][key] = error
        for key in ("observed", "source_calibration_minutes"):
            if not torch.equal(a[key], b[key]):
                raise ValueError(f"Raw control alignment differs: {name}/{key}")
        flags, valid = replay_windows(
            a["energy"].numpy(),
            a["observed"].numpy(),
            old["labels"].numpy(),
            np.zeros(len(old["minutes"]), dtype=int),
            a["source_calibration"].tolist(),
            capacity=buffer,
            policy="continuous",
            fpr=0.05,
        )
        if not np.array_equal(flags, b["flags"].numpy()) or not np.array_equal(
            valid, b["valid"].numpy()
        ):
            raise ValueError(f"Matched-capacity raw decisions differ: {name}")
    report = {
        "control_scores_sha256": sha256(control / "scores.pt"),
        "candidate_scores_sha256": sha256(candidate / "scores.pt"),
        "matched_buffer": buffer,
        "all_control_decisions_exact": True,
        "scaled_energy_errors": errors,
    }
    write_json(candidate / "raw_control_audit.json", report)
    print("raw controls exact at buffer", buffer, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--buffer", type=int, default=39)
    args = parser.parse_args()
    audit(args.control, args.candidate, args.buffer)


if __name__ == "__main__":
    main()
