"""Freeze a model-family consensus comparison on all three datasets."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round227_heterogeneous_consensus_v1", Path(__file__))
for number, dataset in ((227, "cic"), (228, "ton"), (229, "xiiotid")):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.heterogeneous_experiment",
        "--dataset",
        dataset,
        "--output",
        f"New_Model/runs/round{number}_{dataset}_heterogeneous_consensus",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
