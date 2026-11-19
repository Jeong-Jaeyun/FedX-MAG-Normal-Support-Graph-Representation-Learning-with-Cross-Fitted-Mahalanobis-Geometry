"""Match global-vs-entity normal memory without changing feature tensors."""

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


for name, cache, reference, control, number in (
    (
        "preserve",
        "ton_network_quota_node_features_v1.pt",
        "round82_ton_network_quota_nodes",
        "round98_ton_whitened_memory",
        108,
    ),
    (
        "ignore",
        "ton_reliable_node_features_v1.pt",
        "round101_ton_reliable_nodes",
        "round103_ton_reliable_whitened_memory",
        109,
    ),
):
    output = f"New_Model/artifacts/ton_{name}_node_keys_v1"
    run(
        "cache_node_keys",
        "--cache",
        f"New_Model/artifacts/{cache}",
        "--data",
        "New_Model/artifacts/ton_network_quota16_context_v1/records.npz",
        "--keys",
        "New_Model/artifacts/ton_network_quota_entity_keys_v1/keys.npz",
        "--output",
        output,
    )
    run(
        "normal_memory",
        "--cache",
        f"{output}/node_features.pt",
        "--reference-run",
        f"New_Model/runs/{reference}",
        "--global-control",
        f"New_Model/runs/{control}",
        "--output",
        f"New_Model/runs/round{number}_ton_{name}_entity_memory",
        "--workers",
        6,
        "--buffer",
        39,
        "--metric",
        "source_mahalanobis",
        "--scope",
        "entity",
    )
