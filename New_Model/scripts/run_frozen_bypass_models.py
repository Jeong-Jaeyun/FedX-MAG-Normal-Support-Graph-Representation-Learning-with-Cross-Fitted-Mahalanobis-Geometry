"""Isolate raw bypass from neural objective changes on all three datasets."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round144_frozen_raw_bypass_v1", Path(__file__))


def run(module, *arguments):
    command = [sys.executable, "-m", f"fedxmag_next.{module}", *map(str, arguments)]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)


for first, name, data, original, control, aggregation in (
    (
        144,
        "ton",
        "ton_protocol_context_v1",
        "round120_ton_protocol_context_source",
        "round121_ton_protocol_context_flow",
        "top3",
    ),
    (
        146,
        "cic",
        "cic_http_context_v1",
        "round132_cic_dependency_source",
        "round133_cic_dependency_flow",
        "rank_scan",
    ),
    (
        148,
        "xiiotid",
        "xiiotid_context_all_v2",
        "round35_xiiotid_reserved_normal_source",
        "round58_xiiotid_complete_cal_windows",
        "rank_scan",
    ),
):
    source = f"New_Model/runs/round{first}_{name}_frozen_bypass_source"
    flow = f"New_Model/runs/round{first + 1}_{name}_frozen_bypass_flow"
    run(
        "frozen_residual",
        "--data",
        f"New_Model/artifacts/{data}/records.npz",
        "--detector",
        f"New_Model/runs/{original}/source_detector.pt",
        "--output",
        source,
        "--workers",
        6,
    )
    run(
        "window_bridge",
        "--data",
        f"New_Model/artifacts/{data}/records.npz",
        "--detector",
        f"{source}/source_detector.pt",
        "--output",
        flow,
        "--workers",
        6,
        "--buffer",
        39,
        "--history-policy",
        "continuous",
        "--aggregation",
        aggregation,
    )
    run(
        "control_audit",
        "--control",
        f"New_Model/runs/{control}",
        "--candidate",
        flow,
        "--buffer",
        39,
    )
