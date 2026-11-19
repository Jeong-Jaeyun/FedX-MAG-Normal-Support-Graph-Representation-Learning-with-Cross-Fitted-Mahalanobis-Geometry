"""Fixed full evidence factorial after source-normal conditional score scaling."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round225_score_scale_evidence_factorial_v1", Path(__file__))
for number, previous in ((225, 218), (226, 219)):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.score_scale_experiment",
        "--control",
        f"New_Model/runs/round{previous}_cic_current_normal_view",
        "--output",
        f"New_Model/runs/round{number}_cic_score_evidence_factorial",
        "--evidence-factorial",
        "--workers",
        "6",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
