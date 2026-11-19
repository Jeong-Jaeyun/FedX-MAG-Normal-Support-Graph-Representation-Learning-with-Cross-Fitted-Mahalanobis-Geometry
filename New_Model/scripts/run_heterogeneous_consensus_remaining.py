"""Resolve hash-checked parent provenance for the unchanged remaining members."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(
    root, "round230_heterogeneous_consensus_provenance_v2", Path(__file__)
)
for number, dataset in ((230, "ton"), (231, "xiiotid")):
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
