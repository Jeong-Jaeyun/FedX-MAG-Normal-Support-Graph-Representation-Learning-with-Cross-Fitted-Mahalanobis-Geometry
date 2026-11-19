"""Complete the one legacy v3 frontier packet absent from prior full API runs."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round220_legacy_packet_replay_v1", Path(__file__))
command = [
    sys.executable,
    "-m",
    "fedxmag_next.candidate_replay_audit",
    "--registry",
    "New_Model/candidates/prototype_structure_v3/registry.json",
    "--output",
    "New_Model/runs/round220_remaining_packet_replay",
    "--workers",
    "6",
    "--models",
    "xiiotid_buffer50",
]
print("RUN", command, flush=True)
subprocess.run(command, cwd=root, env=environment, check=True)
