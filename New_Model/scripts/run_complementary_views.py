"""Freeze the same complementary-view comparison for each dataset."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round194_complementary_views_v1", Path(__file__))
for number, previous, dataset in ((194, 182, "ton"), (195, 183, "cic"), (196, 184, "xiiotid")):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.complementary_views",
        "--run",
        f"New_Model/runs/round{previous}_{dataset}_service_risk",
        "--output",
        f"New_Model/runs/round{number}_{dataset}_complementary_views",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
