"""Frozen-encoder global/service geometry x upper/two-sided score comparison."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round130_cohort_typicality_v1", Path(__file__))
for first, data, source, control, name in (
    (
        130,
        "ton_role_coverage_context_v1",
        "round111_ton_role_source",
        "round112_ton_role_flow",
        "behavior",
    ),
    (
        131,
        "ton_protocol_context_v1",
        "round120_ton_protocol_context_source",
        "round121_ton_protocol_context_flow",
        "typed",
    ),
):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.cohort_experiment",
        "--data",
        f"New_Model/artifacts/{data}/records.npz",
        "--detector",
        f"New_Model/runs/{source}/source_detector.pt",
        "--control",
        f"New_Model/runs/{control}",
        "--output",
        f"New_Model/runs/round{first}_ton_{name}_cohort_energy",
        "--workers",
        "6",
        "--buffer",
        "39",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
