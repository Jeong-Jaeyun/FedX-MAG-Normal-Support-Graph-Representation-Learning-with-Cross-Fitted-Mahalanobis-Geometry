"""Remove query subsampling, preserving old features and honest HTTP availability."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round153_cic_all_observations_v1", Path(__file__))


def run(module, *arguments):
    command = [sys.executable, "-m", f"fedxmag_next.{module}", *map(str, arguments)]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)


raw = "New_Model/artifacts/cic_all_connections_v1"
context = "New_Model/artifacts/cic_all_context_v1"
data = "New_Model/artifacts/cic_all_legacy_http_v1/records.npz"
run("record_data", "--dataset", "cic", "--denominator", 1, "--workers", 3, "--output", raw)
run("native_context", "--data", f"{raw}/records.npz", "--output", context, "--workers", 3)
run(
    "merge_cic_observations",
    "--data",
    f"{context}/records.npz",
    "--legacy",
    "New_Model/artifacts/cic_http_context_v1/records.npz",
    "--output",
    "New_Model/artifacts/cic_all_legacy_http_v1",
)
template = "New_Model/runs/round132_cic_dependency_source/source_detector.pt"
for number, kind in ((154, "frozen"), (156, "refit")):
    source = f"New_Model/runs/round{number}_cic_all_{kind}_source"
    flow = f"New_Model/runs/round{number + 1}_cic_all_{kind}_flow"
    if kind == "frozen":
        run("expand_frozen_source", "--data", data, "--detector", template, "--output", source)
    else:
        run(
            "source_only",
            "--data",
            data,
            "--output",
            source,
            "--role-template",
            template,
            "--history-gap",
            300,
            "--epochs",
            30,
            "--workers",
            6,
            "--objective",
            "dependency",
        )
    run(
        "window_bridge",
        "--data",
        data,
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
        "window_replay",
        "--run",
        flow,
        "--output",
        f"{flow}/buffer50",
        "--buffer",
        50,
        "--history-policy",
        "continuous",
    )
