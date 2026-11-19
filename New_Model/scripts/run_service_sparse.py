"""Freeze service versus endpoint evidence at39/50/59 on both CIC inputs."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round216_service_sparse_v1", Path(__file__))
for number, previous in ((216, 214), (217, 215)):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.service_sparse_experiment",
        "--control",
        f"New_Model/runs/round{previous}_cic_sparse_top3",
        "--output",
        f"New_Model/runs/round{number}_cic_service_sparse",
        "--workers",
        "6",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
