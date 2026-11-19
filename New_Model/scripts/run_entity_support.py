"""Paired frozen entity support on original and expanded CIC observations."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round158_cic_entity_support_v1", Path(__file__))
for number, data, keys, detector, control in (
    (
        158,
        "cic_http_context_v1",
        "cic_entity_keys_v1",
        "round132_cic_dependency_source",
        "round133_cic_dependency_flow",
    ),
    (
        159,
        "cic_all_legacy_http_v1",
        "cic_all_entity_keys_v1",
        "round156_cic_all_refit_source",
        "round157_cic_all_refit_flow",
    ),
):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.entity_support_experiment",
        "--data",
        f"New_Model/artifacts/{data}/records.npz",
        "--keys",
        f"New_Model/artifacts/{keys}/keys.npz",
        "--detector",
        f"New_Model/runs/{detector}/source_detector.pt",
        "--control",
        f"New_Model/runs/{control}",
        "--output",
        f"New_Model/runs/round{number}_cic_entity_support",
        "--workers",
        "6",
        "--buffer",
        "39",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
