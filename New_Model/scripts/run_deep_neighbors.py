"""Evaluate frozen deep local support first on CIC, then on X-IIoTID."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round150_deep_neighbors_v1", Path(__file__))
for first, name, data, source, control in (
    (
        150,
        "cic",
        "cic_http_context_v1",
        "round132_cic_dependency_source",
        "round133_cic_dependency_flow",
    ),
    (
        151,
        "xiiotid",
        "xiiotid_context_all_v2",
        "round35_xiiotid_reserved_normal_source",
        "round58_xiiotid_complete_cal_windows",
    ),
):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.neighbor_experiment",
        "--data",
        f"New_Model/artifacts/{data}/records.npz",
        "--detector",
        f"New_Model/runs/{source}/source_detector.pt",
        "--control",
        f"New_Model/runs/{control}",
        "--output",
        f"New_Model/runs/round{first}_{name}_deep_neighbors",
        "--workers",
        "6",
        "--buffer",
        "39",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
