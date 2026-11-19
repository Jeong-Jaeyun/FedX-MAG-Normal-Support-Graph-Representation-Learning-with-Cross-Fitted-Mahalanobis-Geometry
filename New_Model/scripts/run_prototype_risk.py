"""Fixed prototype/temporal factorial, TON then full CIC then X-IIoTID."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round170_prototype_risk_v1", Path(__file__))
for number, dataset, origin in ((170, "ton", 165), (171, "cic", 163), (172, "xiiotid", 164)):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.prototype_risk_experiment",
        "--run",
        f"New_Model/runs/round{origin}_{dataset}_causal_risk",
        "--output",
        f"New_Model/runs/round{number}_{dataset}_prototype_risk",
        "--workers",
        "6",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
