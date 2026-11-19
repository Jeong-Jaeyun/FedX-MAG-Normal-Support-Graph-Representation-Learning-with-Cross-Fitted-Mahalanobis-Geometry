"""Follow up the failed quantile-distance study with empirical normal ranks."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round138_empirical_typicality_v1", Path(__file__))
for first, data, source, control, name in (
    (
        138,
        "ton_role_coverage_context_v1",
        "round111_ton_role_source",
        "round112_ton_role_flow",
        "behavior",
    ),
    (
        139,
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
        f"New_Model/runs/round{first}_ton_{name}_empirical_energy",
        "--workers",
        "6",
        "--buffer",
        "39",
        "--score-style",
        "rank",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
