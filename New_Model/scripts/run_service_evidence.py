"""Freeze service routing alone around the completed repeated-evidence test."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round210_service_evidence_v1", Path(__file__))
for number, dataset, normal, previous, evidence, head, view in (
    (212, "xiiotid", 172, 184, 206, "deep_mahalanobis_m8_ensemble", "deep"),
    (211, "cic", 171, 183, 205, "deep_mahalanobis_m1_ensemble", "raw"),
    (210, "ton", 170, 182, 204, "raw_mahalanobis_m1", "raw"),
):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.evidence_experiment",
        "--prototype",
        f"New_Model/runs/round{normal}_{dataset}_prototype_risk",
        "--control",
        f"New_Model/runs/round{previous}_{dataset}_service_risk",
        "--evidence-control",
        f"New_Model/runs/round{evidence}_{dataset}_weak_evidence",
        "--head",
        head,
        "--normal-view",
        view,
        "--routing",
        "service",
        "--output",
        f"New_Model/runs/round{number}_{dataset}_service_evidence",
        "--workers",
        "6",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
