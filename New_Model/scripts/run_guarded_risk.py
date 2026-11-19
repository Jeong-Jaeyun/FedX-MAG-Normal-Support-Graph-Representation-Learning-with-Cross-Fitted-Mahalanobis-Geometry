"""Require current source-normal tail evidence before propagating alarm memory."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round168_cic_guarded_risk_v1", Path(__file__))
for number, origin in ((168, 162), (169, 163)):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.risk_feedback_sensitivity",
        "--run",
        f"New_Model/runs/round{origin}_cic_causal_risk",
        "--output",
        f"New_Model/runs/round{number}_cic_guarded_risk",
        "--current-gate-quantile",
        "0.95",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
