"""Early CIC normal feedback: reference refit vs neural fine-tuning."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round152_cic_early_adaptation_v1", Path(__file__))
command = [
    sys.executable,
    "-m",
    "fedxmag_next.early_adaptation",
    "--data",
    "New_Model/artifacts/cic_http_context_v1/records.npz",
    "--detector",
    "New_Model/runs/round132_cic_dependency_source/source_detector.pt",
    "--control",
    "New_Model/runs/round133_cic_dependency_flow",
    "--output",
    "New_Model/runs/round152_cic_early_normal_adaptation",
    "--workers",
    "6",
]
print("RUN", command, flush=True)
subprocess.run(command, cwd=root, env=environment, check=True)
