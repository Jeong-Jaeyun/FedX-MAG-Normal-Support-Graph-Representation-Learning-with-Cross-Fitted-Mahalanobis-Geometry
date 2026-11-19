"""Freeze a fixed normal-contact topology hypothesis on all three datasets."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round197_normal_topology_v1", Path(__file__))
for number, dataset, normal, previous, head in (
    (197, "ton", 170, 182, "raw_mahalanobis_m1_continuous_prototype_source"),
    (
        198,
        "cic",
        171,
        183,
        "deep_mahalanobis_m1_ensemble_rawview_adaptive_continuous_prototype_risk_fixed_source",
    ),
    (199, "xiiotid", 172, 184, "deep_mahalanobis_m8_ensemble_continuous_prototype_risk_fixed_peer"),
):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.topology_experiment",
        "--prototype",
        f"New_Model/runs/round{normal}_{dataset}_prototype_risk",
        "--control",
        f"New_Model/runs/round{previous}_{dataset}_service_risk",
        "--head",
        head,
        "--output",
        f"New_Model/runs/round{number}_{dataset}_normal_topology",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
