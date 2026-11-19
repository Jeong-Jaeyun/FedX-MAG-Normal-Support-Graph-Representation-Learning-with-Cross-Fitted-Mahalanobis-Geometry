"""Predeclared NeuTraL-style source normal representation and native-score test."""

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


data = "New_Model/artifacts/ton_network_quota16_context_v1/records.npz"
source = "New_Model/runs/round104_ton_transform_source"
detector = f"{source}/source_detector.pt"
flow = "New_Model/runs/round105_ton_transform_flow"
node = "New_Model/runs/round106_ton_transform_nodes"
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
    "--role-order",
    "auto_normal_windows",
    "--constant-feature-policy",
    "preserve",
    "--objective",
    "neural_transformations",
)
run(
    "window_bridge",
    "--data",
    data,
    "--detector",
    detector,
    "--output",
    flow,
    "--workers",
    6,
    "--buffer",
    39,
    "--history-policy",
    "continuous",
    "--aggregation",
    "top3",
)
run(
    "node_experiment",
    "--data",
    data,
    "--detector",
    detector,
    "--output",
    node,
    "--keys",
    "New_Model/artifacts/ton_network_quota_entity_keys_v1/keys.npz",
    "--workers",
    6,
    "--buffer",
    39,
    "--save-features",
    "New_Model/artifacts/ton_transform_node_features_v1.pt",
)
run(
    "joint_rank",
    "--flow",
    flow,
    "--node",
    node,
    "--output",
    "New_Model/runs/round107_ton_transform_joint",
    "--buffer",
    39,
)
