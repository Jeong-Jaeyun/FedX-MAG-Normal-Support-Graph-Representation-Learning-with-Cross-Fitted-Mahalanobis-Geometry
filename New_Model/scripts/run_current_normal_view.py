"""Freeze normality-role separation on old and complete-HTTP CIC."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round218_current_normal_view_v1", Path(__file__))
for number, normal, suffix, previous in (
    (218, 171, "prototype_risk", 216),
    (219, 189, "packet_http_prototype", 217),
):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.current_normal_experiment",
        "--prototype",
        f"New_Model/runs/round{normal}_cic_{suffix}",
        "--control",
        f"New_Model/runs/round{previous}_cic_service_sparse",
        "--output",
        f"New_Model/runs/round{number}_cic_current_normal_view",
        "--workers",
        "6",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
