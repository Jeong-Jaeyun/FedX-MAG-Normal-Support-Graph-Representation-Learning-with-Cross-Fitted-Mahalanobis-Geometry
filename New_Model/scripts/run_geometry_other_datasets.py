"""Matched direct-geometry replication on CIC and X-IIoTID development data."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round132_cross_dataset_geometry_v1", Path(__file__))


def run(module, *arguments):
    command = [sys.executable, "-m", f"fedxmag_next.{module}", *map(str, arguments)]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)


for number, name, data, objective, template, control in (
    (132, "cic_dependency", "cic_http_context_v1", "dependency", None, None),
    (
        134,
        "cic_geometry",
        "cic_http_context_v1",
        "geometry_margin",
        "round132_cic_dependency_source",
        "round133_cic_dependency_flow",
    ),
    (
        136,
        "xiiotid_geometry",
        "xiiotid_context_all_v2",
        "geometry_margin",
        "round35_xiiotid_reserved_normal_source",
        "round58_xiiotid_complete_cal_windows",
    ),
):
    source = f"New_Model/runs/round{number}_{name}_source"
    flow = f"New_Model/runs/round{number + 1}_{name}_flow"
    extra = ("--role-template", f"New_Model/runs/{template}/source_detector.pt") if template else ()
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
        objective,
        "--history-gap",
        300,
        "--constant-feature-policy",
        "preserve",
        *extra,
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
    if control is None:
        continue
    old = torch.load(root / f"New_Model/runs/{control}/scores.pt", weights_only=True)
    new = torch.load(root / f"{flow}/scores.pt", weights_only=True)
    for key in ("minutes", "labels"):
        if not torch.equal(old[key], new[key]):
            raise ValueError(f"Control windows/labels differ: {key}")
    # Old X control used buffer50. Recompute its same frozen energies at39,
    # explicitly; never compare its stored50 decisions with new39 decisions.
    sys.path.insert(0, environment["PYTHONPATH"])
    from fedxmag_next.window_bridge import replay_windows

    errors = {}
    for method in (
        "raw_knn",
        "raw_mahalanobis_m1",
        "raw_mahalanobis_m8",
        "record_count",
        "sampled_record_count",
    ):
        a, b = old["methods"][method], new["methods"][method]
        errors[method] = {}
        for key in ("energy", "source_calibration"):
            error = float(((a[key] - b[key]).abs() / (1 + a[key].abs())).max())
            if error > 1e-8:
                raise ValueError(f"Matched raw score differs: {method}/{key}/{error}")
            errors[method][key] = error
        for key in ("observed", "source_calibration_minutes"):
            if not torch.equal(a[key], b[key]):
                raise ValueError(f"Control alignment differs: {method}/{key}")
        flags, valid = replay_windows(
            a["energy"].numpy(),
            a["observed"].numpy(),
            old["labels"].numpy(),
            np.zeros(len(old["minutes"]), dtype=int),
            a["source_calibration"].tolist(),
            capacity=39,
            policy="continuous",
            fpr=0.05,
        )
        if not np.array_equal(flags, b["flags"].numpy()) or not np.array_equal(
            valid, b["valid"].numpy()
        ):
            raise ValueError(f"Matched buffer39 control decisions differ: {method}")
    with (root / f"{flow}/raw_control_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "matched_buffer": 39,
                "all_control_decisions_exact": True,
                "scaled_energy_errors": errors,
            },
            stream,
            indent=2,
        )
