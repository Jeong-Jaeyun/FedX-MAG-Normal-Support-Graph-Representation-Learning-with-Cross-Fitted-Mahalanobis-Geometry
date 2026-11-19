"""Freeze and execute complete packet HTTP acquisition and matched-role refits."""

import os
import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
snapshot_name = "round185_full_http_v1"
environment = frozen_environment(
    root,
    snapshot_name,
    Path(__file__),
    extra_files={
        f"tracker_snapshot/fedxmag/{path.name}": path
        for path in sorted((root / "src/fedxmag").glob("*.py"))
    },
)
tracker = root / "New_Model/experiment_sources" / snapshot_name / "tracker_snapshot"
environment["PYTHONPATH"] += os.pathsep + str(tracker)
data = "New_Model/artifacts/cic_all_packet_http_v1/records.npz"
source = "New_Model/runs/round186_cic_packet_http_source/source_detector.pt"
flow = "New_Model/runs/round187_cic_packet_http_flow"
risk = "New_Model/runs/round188_cic_packet_http_risk"
prototype = "New_Model/runs/round189_cic_packet_http_prototype"
continuous = "New_Model/runs/round190_cic_packet_http_continuous"
calibration = "New_Model/runs/round191_cic_packet_http_calibration"
commands = [
    (
        "full_http_observations",
        [
            "--data",
            "New_Model/artifacts/cic_all_legacy_http_v1/records.npz",
            "--output",
            "New_Model/artifacts/cic_all_packet_http_v1",
            "--tracker-root",
            str(tracker),
            "--workers",
            "3",
        ],
    ),
    (
        "source_only",
        [
            "--data",
            data,
            "--output",
            str(Path(source).parent),
            "--role-template",
            "New_Model/runs/round156_cic_all_refit_source/source_detector.pt",
            "--workers",
            "6",
            "--epochs",
            "30",
            "--history-gap",
            "300",
        ],
    ),
    (
        "window_bridge",
        [
            "--data",
            data,
            "--detector",
            source,
            "--output",
            flow,
            "--workers",
            "6",
            "--buffer",
            "39",
            "--history-policy",
            "continuous",
            "--aggregation",
            "rank_scan",
        ],
    ),
    (
        "window_replay",
        ["--run", flow, "--output", flow + "/capacity50", "--buffer", "50"],
    ),
    (
        "risk_experiment",
        [
            "--data",
            data,
            "--keys",
            "New_Model/artifacts/cic_all_entity_keys_v1/keys.npz",
            "--detector",
            source,
            "--control",
            flow,
            "--output",
            risk,
            "--workers",
            "6",
        ],
    ),
    ("prototype_risk_experiment", ["--run", risk, "--output", prototype, "--workers", "6"]),
    ("prototype_continuous", ["--run", prototype, "--output", continuous, "--workers", "6"]),
    ("prototype_calibration", ["--run", continuous, "--output", calibration]),
    (
        "prototype_cross_view",
        [
            "--run",
            prototype,
            "--control",
            calibration,
            "--output",
            "New_Model/runs/round192_cic_packet_http_cross_view",
            "--workers",
            "6",
        ],
    ),
]
for module, arguments in commands:
    command = [sys.executable, "-m", f"fedxmag_next.{module}", *arguments]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
