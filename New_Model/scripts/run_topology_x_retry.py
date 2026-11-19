"""Retry X with its unchanged calibration-selection-train source-role order."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round200_x_topology_role_audit_v1", Path(__file__))
command = [
    sys.executable,
    "-m",
    "fedxmag_next.topology_experiment",
    "--prototype",
    "New_Model/runs/round172_xiiotid_prototype_risk",
    "--control",
    "New_Model/runs/round184_xiiotid_service_risk",
    "--head",
    "deep_mahalanobis_m8_ensemble_continuous_prototype_risk_fixed_peer",
    "--output",
    "New_Model/runs/round200_xiiotid_normal_topology",
]
print("RUN", command, flush=True)
subprocess.run(command, cwd=root, env=environment, check=True)
