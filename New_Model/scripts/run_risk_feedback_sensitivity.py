"""Ablate normal-feedback risk clearing without changing model or gate parameters."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round166_cic_risk_no_clear_v1", Path(__file__))
for number, origin in ((166, 162), (167, 163)):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.risk_feedback_sensitivity",
        "--run",
        f"New_Model/runs/round{origin}_cic_causal_risk",
        "--output",
        f"New_Model/runs/round{number}_cic_risk_no_clear",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
