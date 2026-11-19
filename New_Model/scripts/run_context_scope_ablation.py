"""Predeclared matched TON context ablations; stop immediately on failure."""

import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
os.chdir(root)
os.environ["PYTHONPATH"] = str(root / "New_Model/src")
os.environ["PYTHONIOENCODING"] = "utf-8"
base = "New_Model/artifacts/ton_network_quota16_context_v1/records.npz"
keys = "New_Model/artifacts/ton_network_quota_entity_keys_v1/keys.npz"


def run(module, *arguments):
    command = [sys.executable, "-m", f"fedxmag_next.{module}", *map(str, arguments)]
    print("RUN", command, flush=True)
    subprocess.run(command, check=True)


for policy, first in (("current", 85), ("pair", 89), ("outgoing_pair", 93)):
    data_directory = f"New_Model/artifacts/ton_network_quota_{policy}_v1"
    source = f"New_Model/runs/round{first}_ton_{policy}_source"
    flow = f"New_Model/runs/round{first + 1}_ton_{policy}_flow"
    nodes = f"New_Model/runs/round{first + 2}_ton_{policy}_nodes"
    joint = f"New_Model/runs/round{first + 3}_ton_{policy}_joint"
    run("feature_subset", "--data", base, "--output", data_directory, "--policy", policy)
    data = f"{data_directory}/records.npz"
    detector = f"{source}/source_detector.pt"
    run(
        "source_only",
        "--data",
        data,
        "--output",
        source,
        "--epochs",
        30,
        "--workers",
        6,
        "--role-order",
        "auto_normal_windows",
    )
    run(
        "window_bridge",
        "--data",
        data,
        "--detector",
        detector,
        "--output",
        flow,
        "--buffer",
        39,
        "--workers",
        6,
        "--history-policy",
        "continuous",
        "--aggregation",
        "top3",
    )
    run(
        "node_experiment",
        "--data",
        data,
        "--keys",
        keys,
        "--detector",
        detector,
        "--output",
        nodes,
        "--buffer",
        39,
        "--workers",
        6,
    )
    run(
        "joint_rank",
        "--flow",
        flow,
        "--node",
        nodes,
        "--output",
        joint,
        "--buffer",
        39,
    )
