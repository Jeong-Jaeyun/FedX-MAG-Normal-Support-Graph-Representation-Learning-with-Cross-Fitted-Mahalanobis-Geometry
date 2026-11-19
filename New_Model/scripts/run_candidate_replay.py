"""Freeze the integrated initial-state API audit independently of packet work."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round193_candidate_replay_v1", Path(__file__))
command = [
    sys.executable,
    "-m",
    "fedxmag_next.candidate_replay_audit",
    "--registry",
    "New_Model/candidates/prototype_structure_v2/registry.json",
    "--output",
    "New_Model/runs/round193_exported_candidate_replay",
    "--workers",
    "6",
    "--models",
    "ton_buffer39",
    "cic_buffer39",
    "cic_buffer50",
    "xiiotid_buffer39",
    "xiiotid_service_buffer50",
]
print("RUN", command, flush=True)
subprocess.run(command, cwd=root, env=environment, check=True)
