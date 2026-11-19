"""Test outgoing protocol history with immutable execution sources."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round120_protocol_context_v1", Path(__file__))


def run(module, *arguments):
    command = [sys.executable, "-m", f"fedxmag_next.{module}", *map(str, arguments)]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)


data = "New_Model/artifacts/ton_protocol_context_v1/records.npz"
source = "New_Model/runs/round120_ton_protocol_context_source"
run(
    "source_only",
    "--data",
    data,
    "--output",
    source,
    "--workers",
    6,
    "--epochs",
    30,
    "--role-template",
    "New_Model/runs/round118_ton_protocol_source/source_detector.pt",
    "--allow-template-feature-change",
    "--history-gap",
    300,
    "--constant-feature-policy",
    "preserve",
)
run(
    "window_bridge",
    "--data",
    data,
    "--detector",
    f"{source}/source_detector.pt",
    "--output",
    "New_Model/runs/round121_ton_protocol_context_flow",
    "--workers",
    6,
    "--buffer",
    39,
    "--history-policy",
    "continuous",
    "--aggregation",
    "top3",
)
