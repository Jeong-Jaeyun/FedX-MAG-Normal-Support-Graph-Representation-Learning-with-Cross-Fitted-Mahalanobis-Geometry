"""Source-only input reliability and covariance-aware normal-memory hypotheses."""

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


run(
    "normal_memory",
    "--cache",
    "New_Model/artifacts/ton_network_quota_node_features_v1.pt",
    "--reference-run",
    "New_Model/runs/round82_ton_network_quota_nodes",
    "--output",
    "New_Model/runs/round98_ton_whitened_memory",
    "--workers",
    6,
    "--buffer",
    39,
    "--metric",
    "source_mahalanobis",
)
data = "New_Model/artifacts/ton_network_quota16_context_v1/records.npz"
source = "New_Model/runs/round99_ton_reliable_source"
detector = f"{source}/source_detector.pt"
flow = "New_Model/runs/round100_ton_reliable_flow"
node = "New_Model/runs/round101_ton_reliable_nodes"
cache = "New_Model/artifacts/ton_reliable_node_features_v1.pt"
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
    "ignore",
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
    cache,
)
run(
    "joint_rank",
    "--flow",
    flow,
    "--node",
    node,
    "--output",
    "New_Model/runs/round102_ton_reliable_joint",
    "--buffer",
    39,
)
run(
    "normal_memory",
    "--cache",
    cache,
    "--reference-run",
    node,
    "--output",
    "New_Model/runs/round103_ton_reliable_whitened_memory",
    "--workers",
    6,
    "--buffer",
    39,
    "--metric",
    "source_mahalanobis",
)
