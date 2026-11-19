"""Matched context negatives vs feature-group negatives, with/without typed history."""

import json
import subprocess
import sys
from pathlib import Path

import torch
from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round122_context_association_v1", Path(__file__))


def run(module, *arguments):
    command = [sys.executable, "-m", f"fedxmag_next.{module}", *map(str, arguments)]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)


for name, data, template, control, first in (
    (
        "typed",
        "ton_protocol_context_v1",
        "round120_ton_protocol_context_source",
        "round121_ton_protocol_context_flow",
        122,
    ),
    (
        "application",
        "ton_protocol_fields_v1",
        "round118_ton_protocol_source",
        "round119_ton_protocol_flow",
        124,
    ),
):
    source = f"New_Model/runs/round{first}_ton_{name}_association_source"
    flow = f"New_Model/runs/round{first + 1}_ton_{name}_association_flow"
    run(
        "source_only",
        "--data",
        f"New_Model/artifacts/{data}/records.npz",
        "--output",
        source,
        "--workers",
        6,
        "--epochs",
        30,
        "--objective",
        "association",
        "--role-template",
        f"New_Model/runs/{template}/source_detector.pt",
        "--history-gap",
        300,
        "--constant-feature-policy",
        "preserve",
    )
    run(
        "window_bridge",
        "--data",
        f"New_Model/artifacts/{data}/records.npz",
        "--detector",
        f"{source}/source_detector.pt",
        "--output",
        flow,
        "--workers",
        6,
        "--buffer",
        39,
        "--history-policy",
        "continuous",
        "--aggregation",
        "top3",
    )
    old = torch.load(root / f"New_Model/runs/{control}/scores.pt", weights_only=True)
    new = torch.load(root / f"{flow}/scores.pt", weights_only=True)
    errors = {}
    for method in (
        "raw_knn",
        "raw_mahalanobis_m1",
        "raw_mahalanobis_m8",
        "record_count",
        "sampled_record_count",
    ):
        errors[method] = {}
        for key in ("energy", "source_calibration"):
            a, b = old["methods"][method][key], new["methods"][method][key]
            error = float(((a - b).abs() / (1 + a.abs())).max())
            if error > 1e-8:
                raise ValueError(f"Matched raw control differs: {method}/{key}: {error}")
            errors[method][key] = error
        for key in ("flags", "valid", "source_calibration_minutes"):
            if not torch.equal(old["methods"][method][key], new["methods"][method][key]):
                raise ValueError(f"Matched raw control decision/alignment differs: {method}/{key}")
    with (root / f"{flow}/raw_control_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {"all_control_decisions_exact": True, "scaled_energy_errors": errors}, stream, indent=2
        )
