"""One calibrated alarm from fixed complementary causal energy channels."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .experiments import sha256, write_json
from .joint_rank import joint_rank_pvalue

RAW = "raw_mahalanobis_m1_continuous_prototype_source"
RAW_DEEP = "deep_mahalanobis_m1_ensemble_rawview_continuous_prototype_risk_fixed_source"
DEEP_PEER = "deep_mahalanobis_m8_ensemble_continuous_prototype_risk_fixed_peer"
DEEP_SOURCE = "deep_mahalanobis_m8_ensemble_continuous_prototype_risk_fixed_source"
DESIGNS = {
    "raw1_and_deep8peer": [RAW, DEEP_PEER],
    "rawview_deep1_and_deep8peer": [RAW_DEEP, DEEP_PEER],
    "raw1_rawview_deep1_deep8peer": [RAW, RAW_DEEP, DEEP_PEER],
    "deep8_source_and_peer": [DEEP_SOURCE, DEEP_PEER],
}


def replay(calibration, queries, feedback, observed, labels, capacity):
    history = deque(calibration[-capacity:].tolist(), maxlen=capacity)
    pvalues, valid = np.ones(len(queries)), np.zeros(len(queries), dtype=bool)
    for i, query in enumerate(queries):
        if observed[i] and len(history) >= 19:
            pvalues[i] = joint_rank_pvalue(np.asarray(history), query)
            valid[i] = True
        if observed[i] and labels[i] == 0:
            history.append(feedback[i].tolist())
    return valid & (pvalues <= 0.05), valid, pvalues


def metrics(flags, valid, labels, types):
    return {
        "coverage": float(valid.mean()),
        "attack_detected": int(flags[labels == 1].sum()),
        "attack_windows": int((labels == 1).sum()),
        "normal_alarms": int(flags[labels == 0].sum()),
        "normal_windows": int((labels == 0).sum()),
        "tpr": float(flags[labels == 1].mean()),
        "fpr": float(flags[labels == 0].mean()),
        "per_type_tpr": {
            kind: {
                "windows": sum(kind in group for group in types),
                "tpr": float(flags[[kind in group for group in types]].mean()),
            }
            for kind in sorted({kind for group in types for kind in group})
        },
    }


def run(args):
    origin, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    registration = json.loads((origin / "registration.json").read_text())
    for capacity in (39, 50):
        source = origin if capacity == 39 else origin / "capacity50"
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir()
        archive = torch.load(source / "scores.pt", weights_only=True)
        old_decisions = torch.load(source / "evaluation/decisions.pt", weights_only=True)
        labels = archive["labels"].numpy()
        components = sorted({name for names in DESIGNS.values() for name in names})
        common = None
        for name in components:
            local = archive["methods"][name]["source_calibration_minutes"].numpy()
            common = local if common is None else np.intersect1d(common, local)
        if len(common) < 19:
            raise ValueError("Not enough common independent source-normal windows")
        cal, query, feedback_base = {}, {}, {}
        for name in components:
            item = archive["methods"][name]
            times = item["source_calibration_minutes"].numpy()
            indices = np.searchsorted(times, common)
            if not np.array_equal(times[indices], common):
                raise ValueError("Complementary source-calibration windows do not align")
            cal[name] = item["source_calibration"].numpy()[indices]
            query[name] = item["energy"].numpy()
            base_head = name.split("_continuous_")[0].replace("_rawview", "")
            feedback_base[name] = archive["methods"][base_head]["energy"].numpy()
            for mode in ("frozen", "candidate"):
                feedback = feedback_base[name] if mode == "frozen" else query[name]
                flags, valid, _ = replay(
                    cal[name][:, None],
                    query[name][:, None],
                    feedback[:, None],
                    item["observed"].numpy(),
                    labels,
                    capacity,
                )
                expected_name = (
                    name
                    if mode == "frozen"
                    else name.replace("_continuous_", "_adaptive_continuous_", 1)
                )
                expected = old_decisions[expected_name]
                if not np.array_equal(flags, expected["flags"].numpy()) or not np.array_equal(
                    valid, expected["valid"].numpy()
                ):
                    raise ValueError(f"Single-channel matched control changed: {expected_name}")
        reports, predictions = {}, {}
        for design, names in DESIGNS.items():
            observed = np.logical_and.reduce(
                [archive["methods"][name]["observed"].numpy() for name in names]
            )
            matrix = np.column_stack([query[name] for name in names])
            calibration = np.column_stack([cal[name] for name in names])
            for mode in ("frozen", "candidate"):
                feedback = (
                    np.column_stack([feedback_base[name] for name in names])
                    if mode == "frozen"
                    else matrix
                )
                flags, valid, pvalues = replay(
                    calibration, matrix, feedback, observed, labels, capacity
                )
                name = f"{design}_{mode}"
                reports[name] = metrics(flags, valid, labels, archive["types"])
                predictions[name] = {
                    "flags": torch.from_numpy(flags),
                    "valid": torch.from_numpy(valid),
                    "pvalue": torch.from_numpy(pvalues),
                }
                print(capacity, name, {k: reports[name][k] for k in ("tpr", "fpr")}, flush=True)
        write_json(
            destination / "report.json",
            {
                "status": "fixed_complementary_views_development",
                "dataset": registration["dataset"],
                "capacity": capacity,
                "all_single_channel_parent_decisions_exact": True,
                "methods": reports,
            },
        )
        write_json(
            destination / "registration.json",
            {
                "args": vars(args),
                "scores_sha256": sha256(source / "scores.pt"),
                "designs": DESIGNS,
                "calibration_minutes": common.tolist(),
                "policy": (
                    "symmetric min-upper-rank, p<=.05; predict before verified-normal feedback"
                ),
                "capacity": capacity,
            },
        )
        torch.save(
            {
                "minutes": archive["minutes"],
                "labels": archive["labels"],
                "types": archive["types"],
                "methods": predictions,
            },
            destination / "decisions.pt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
