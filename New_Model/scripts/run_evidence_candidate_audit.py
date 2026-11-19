"""Freeze a full initial-state API replay of the new X95%/5% candidate."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round213_evidence_api_replay_v1", Path(__file__))
command = [
    sys.executable,
    "-m",
    "fedxmag_next.candidate_replay_audit",
    "--registry",
    "New_Model/candidates/prototype_structure_v3/registry.json",
    "--output",
    "New_Model/runs/round213_evidence_candidate_replay",
    "--workers",
    "6",
    "--models",
    "xiiotid_evidence_buffer50",
]
print("RUN", command, flush=True)
subprocess.run(command, cwd=root, env=environment, check=True)
