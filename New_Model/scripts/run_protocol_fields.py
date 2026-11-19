"""Same observations/physical roles, adding generic wire/application metadata."""

import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
os.chdir(root)
os.environ["PYTHONPATH"] = str(root / "New_Model/src")
os.environ["PYTHONIOENCODING"] = "utf-8"


def run(module, *arguments):
    command = [sys.executable, "-m", f"fedxmag_next.{module}", *map(str, arguments)]
    print("RUN", command, flush=True)
    subprocess.run(command, check=True)


data = "New_Model/artifacts/ton_protocol_fields_v1/records.npz"
source = "New_Model/runs/round118_ton_protocol_source"
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
    "New_Model/runs/round111_ton_role_source/source_detector.pt",
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
    "New_Model/runs/round119_ton_protocol_flow",
    "--workers",
    6,
    "--buffer",
    39,
    "--history-policy",
    "continuous",
    "--aggregation",
    "top3",
)
