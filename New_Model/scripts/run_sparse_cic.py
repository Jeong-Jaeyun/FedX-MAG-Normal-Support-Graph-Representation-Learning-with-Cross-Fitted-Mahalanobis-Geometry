"""Freeze sparse pooling on both old and complete-HTTP CIC inputs."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round214_sparse_cic_v1", Path(__file__))
for number, normal, suffix in ((214, 171, "prototype_risk"), (215, 189, "packet_http_prototype")):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.sparse_cic_experiment",
        "--prototype",
        f"New_Model/runs/round{normal}_cic_{suffix}",
        "--output",
        f"New_Model/runs/round{number}_cic_sparse_top3",
        "--workers",
        "6",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
