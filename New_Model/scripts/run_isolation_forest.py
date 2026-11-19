"""Freeze a stronger raw normal-support comparison for all three datasets."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round207_isolation_forest_v1", Path(__file__))
for number, dataset, normal, suffix in (
    (207, "ton", 170, "prototype_risk"),
    (208, "cic", 189, "packet_http_prototype"),
    (209, "xiiotid", 172, "prototype_risk"),
):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.forest_experiment",
        "--prototype",
        f"New_Model/runs/round{normal}_{dataset}_{suffix}",
        "--output",
        f"New_Model/runs/round{number}_{dataset}_isolation_forest",
        "--workers",
        "6",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
