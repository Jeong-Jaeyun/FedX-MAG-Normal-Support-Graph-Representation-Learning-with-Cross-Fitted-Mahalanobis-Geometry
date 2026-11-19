"""Refine temporal state by observed service while preserving normal geometry."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round182_service_risk_v1", Path(__file__))
for number, dataset, normal, control in (
    (182, "ton", 170, 179),
    (183, "cic", 171, 180),
    (184, "xiiotid", 172, 181),
):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.prototype_cross_view",
        "--run",
        f"New_Model/runs/round{normal}_{dataset}_prototype_risk",
        "--control",
        f"New_Model/runs/round{control}_{dataset}_cross_view",
        "--output",
        f"New_Model/runs/round{number}_{dataset}_service_risk",
        "--workers",
        "6",
        "--intervention",
        "service_keys",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
