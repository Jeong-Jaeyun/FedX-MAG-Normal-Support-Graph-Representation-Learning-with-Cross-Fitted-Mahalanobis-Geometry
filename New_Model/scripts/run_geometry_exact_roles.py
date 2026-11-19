"""Repeat the invalid CIC comparison and X replication with exact saved roles."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round140_exact_role_geometry_v1", Path(__file__))


def run(module, *arguments):
    command = [sys.executable, "-m", f"fedxmag_next.{module}", *map(str, arguments)]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)


for first, name, data, template, control in (
    (
        140,
        "cic",
        "cic_http_context_v1",
        "round132_cic_dependency_source",
        "round133_cic_dependency_flow",
    ),
    (
        142,
        "xiiotid",
        "xiiotid_context_all_v2",
        "round35_xiiotid_reserved_normal_source",
        "round58_xiiotid_complete_cal_windows",
    ),
):
    source = f"New_Model/runs/round{first}_{name}_exact_geometry_source"
    flow = f"New_Model/runs/round{first + 1}_{name}_exact_geometry_flow"
    run(
        "source_only",
        "--data",
        f"New_Model/artifacts/{data}/records.npz",
        "--output",
        source,
        "--workers",
        6,
        "--epochs",
        30,
        "--objective",
        "geometry_margin",
        "--role-template",
        f"New_Model/runs/{template}/source_detector.pt",
        "--history-gap",
        300,
        "--constant-feature-policy",
        "preserve",
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
        "rank_scan",
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
