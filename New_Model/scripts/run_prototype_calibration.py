"""Freeze one final-calibration ablation after the continuous parent completes."""

import argparse
import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--round", type=int, required=True)
parser.add_argument("--parent", type=int, required=True)
parser.add_argument("--dataset", choices=("ton", "cic", "xiiotid"), required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
parent = f"New_Model/runs/round{args.parent}_{args.dataset}_continuous_prototype"
if not (root / parent / "capacity50/report.json").exists():
    raise ValueError("Continuous parent has not completed both capacities")
environment = frozen_environment(
    root, f"round{args.round}_prototype_calibration_v1", Path(__file__)
)
command = [
    sys.executable,
    "-m",
    "fedxmag_next.prototype_calibration",
    "--run",
    parent,
    "--output",
    f"New_Model/runs/round{args.round}_{args.dataset}_prototype_calibration",
]
print("RUN", command, flush=True)
subprocess.run(command, cwd=root, env=environment, check=True)
