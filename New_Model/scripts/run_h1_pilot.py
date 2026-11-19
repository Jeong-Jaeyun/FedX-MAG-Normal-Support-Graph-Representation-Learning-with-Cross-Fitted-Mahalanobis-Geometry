"""Freeze H1 sources and execute every dataset with one preregistered policy."""

import os
import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
env = frozen_environment(
    root,
    "round233_h1_projected_pilot_v2",
    Path(__file__),
    extra_files={"protocol.md": root / "New_Model/docs/H1_PILOT_PROTOCOL_2026-11-15.md"},
)
env["PATH"] = str(Path(sys.prefix) / "Library/bin") + os.pathsep + env["PATH"]
env["MKL_THREADING_LAYER"] = "SEQUENTIAL"
for dataset in ("cic", "xiiotid", "ton"):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.h1_experiment",
        "--root",
        str(root),
        "--dataset",
        dataset,
        "--output",
        f"New_Model/runs/round233_h1_{dataset}_pilot_v2",
    ]
    subprocess.run(command, cwd=root, env=env, check=True)
