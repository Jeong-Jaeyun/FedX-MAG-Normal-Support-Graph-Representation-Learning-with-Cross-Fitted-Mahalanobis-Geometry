"""Same physical source intervals; add observations covering all directed roles."""

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


base = "New_Model/artifacts/ton_role_coverage_context_v1"
data = f"{base}/records.npz"
source = "New_Model/runs/round111_ton_role_source"
detector = f"{source}/source_detector.pt"
flow = "New_Model/runs/round112_ton_role_flow"
nodes = "New_Model/runs/round113_ton_role_nodes"
keys = "New_Model/artifacts/ton_role_entity_keys_v1"
cache = "New_Model/artifacts/ton_role_node_features_v1.pt"
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
    "New_Model/runs/round80_ton_network_quota_source/source_detector.pt",
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
run("entity_keys", "--data", data, "--native-run", base, "--output", keys)
run(
    "node_experiment",
    "--data",
    data,
    "--detector",
    detector,
    "--output",
    nodes,
    "--keys",
    f"{keys}/keys.npz",
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
    nodes,
    "--output",
    "New_Model/runs/round114_ton_role_joint",
    "--buffer",
    39,
)
run(
    "normal_memory",
    "--cache",
    cache,
    "--reference-run",
    nodes,
    "--output",
    "New_Model/runs/round115_ton_role_global_memory",
    "--workers",
    6,
    "--buffer",
    39,
    "--metric",
    "source_mahalanobis",
    "--scope",
    "global",
)
run(
    "normal_memory",
    "--cache",
    cache,
    "--reference-run",
    nodes,
    "--global-control",
    "New_Model/runs/round115_ton_role_global_memory",
    "--output",
    "New_Model/runs/round116_ton_role_entity_memory",
    "--workers",
    6,
    "--buffer",
    39,
    "--metric",
    "source_mahalanobis",
    "--scope",
    "entity",
)
run(
    "trusted_adjacency",
    "--data",
    data,
    "--keys",
    f"{keys}/keys.npz",
    "--detector",
    detector,
    "--control",
    flow,
    "--output",
    "New_Model/runs/round117_ton_role_trusted_adjacency",
    "--buffer",
    39,
)
