"""Freeze raw-normality/deep-energy comparison for all three datasets."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round179_cross_view_prototypes_v1", Path(__file__))
for number, dataset, normal, control in (
    (179, "ton", 170, 176),
    (180, "cic", 171, 177),
    (181, "xiiotid", 172, 178),
):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.prototype_cross_view",
        "--run",
        f"New_Model/runs/round{normal}_{dataset}_prototype_risk",
        "--control",
        f"New_Model/runs/round{control}_{dataset}_prototype_calibration",
        "--output",
        f"New_Model/runs/round{number}_{dataset}_cross_view",
        "--workers",
        "6",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
