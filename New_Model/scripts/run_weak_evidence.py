"""Freeze peak/sum and strong/weak initiation for all three datasets."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round201_weak_evidence_v1", Path(__file__))
for number, dataset, normal, previous, head, view in (
    (201, "ton", 170, 182, "raw_mahalanobis_m1", "raw"),
    (202, "cic", 171, 183, "deep_mahalanobis_m1_ensemble", "raw"),
    (203, "xiiotid", 172, 184, "deep_mahalanobis_m8_ensemble", "deep"),
):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.evidence_experiment",
        "--prototype",
        f"New_Model/runs/round{normal}_{dataset}_prototype_risk",
        "--control",
        f"New_Model/runs/round{previous}_{dataset}_service_risk",
        "--head",
        head,
        "--normal-view",
        view,
        "--output",
        f"New_Model/runs/round{number}_{dataset}_weak_evidence",
        "--workers",
        "6",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
