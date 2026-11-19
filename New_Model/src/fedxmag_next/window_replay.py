"""Replay normal-window feedback without retraining or changing query energies."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .experiments import sha256, write_json
from .record_calibration import cluster_difference
from .window_bridge import replay_windows


def run(args: argparse.Namespace) -> None:
    source, output = Path(args.run).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = torch.load(source / "scores.pt", map_location="cpu", weights_only=True)
    minutes, labels = data["minutes"].numpy(), data["labels"].numpy()
    origin_path = source / "report.json"
    if not origin_path.exists():
        origin_path = source / "registration.json"
    origin_report = json.loads(origin_path.read_text())
    dataset = origin_report.get("dataset")
    offset = {"ton": 36000, "cic": -10800, "xiiotid": 0}.get(dataset, 0)
    captures = (minutes * 60 + offset) // 86400
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "scores_sha256": sha256(source / "scores.pt"),
            "origin_registration_sha256": sha256(source / "registration.json"),
            "fit": "none; frozen query energies; verified normal windows only update thresholds",
            "bootstrap_unit": "whole observed UTC hour, conditional development uncertainty",
        },
    )
    results, flags_all, valid_all, archive = {}, {}, {}, {}
    for name, stored in data["methods"].items():
        if "source_calibration" not in stored:
            raise ValueError("Window run predates persisted independent source calibration")
        energy, observed = stored["energy"].numpy(), stored["observed"].numpy()
        flags, valid = replay_windows(
            energy,
            observed,
            labels,
            captures,
            stored["source_calibration"].tolist(),
            capacity=args.buffer,
            policy=args.history_policy,
            fpr=args.fpr,
            feedback_energy=stored.get("feedback_energy", stored["energy"]).numpy(),
        )
        normal, attack = valid & (labels == 0), valid & (labels == 1)
        result = {
            "coverage": float(valid.mean()),
            "evaluated_windows": int(valid.sum()),
            "normal_windows": int(normal.sum()),
            "attack_windows": int(attack.sum()),
            "tpr": float(flags[attack].mean()) if attack.any() else None,
            "fpr": float(flags[normal].mean()) if normal.any() else None,
            "end_to_end_attack_recall": float(flags[labels == 1].mean()),
            "auroc": float(roc_auc_score(labels[valid], energy[valid]))
            if normal.any() and attack.any()
            else None,
        }
        result["per_capture"] = {}
        for capture in np.unique(captures):
            group = captures == capture
            normals, attacks = normal & group, attack & group
            result["per_capture"][str(int(capture))] = {
                "coverage": float(valid[group].mean()),
                "normal_windows": int(normals.sum()),
                "attack_windows": int(attacks.sum()),
                "fpr": float(flags[normals].mean()) if normals.any() else None,
                "tpr": float(flags[attacks].mean()) if attacks.any() else None,
            }
        kinds = sorted({kind for group in data["types"] for kind in group})
        result["per_type_tpr"] = {}
        for kind in kinds:
            members = np.asarray([kind in group for group in data["types"]])
            chosen = members & valid
            result["per_type_tpr"][kind] = {
                "available_windows": int(members.sum()),
                "evaluated_windows": int(chosen.sum()),
                "tpr": float(flags[chosen].mean()) if chosen.any() else None,
                "end_to_end_recall": float(flags[members].mean()),
            }
        results[name], flags_all[name], valid_all[name] = result, flags, valid
        archive[name] = {"flags": torch.from_numpy(flags), "valid": torch.from_numpy(valid)}
        print(name, {key: result[key] for key in ("tpr", "fpr", "coverage")}, flush=True)
    differences = {
        name: cluster_difference(
            (flags_all[name], valid_all[name]),
            (flags_all["raw_knn"], valid_all["raw_knn"]),
            labels,
            minutes // 60,
        )
        for name in results
        if name.startswith(("deep_", "raw_plus_"))
    }
    write_json(
        output / "report.json",
        {
            "status": "frozen_energy_window_replay_development",
            "dataset": dataset,
            "methods": results,
            "hour_bootstrap_vs_knn": differences,
        },
    )
    torch.save(archive, output / "decisions.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--buffer", type=int, default=20)
    parser.add_argument("--fpr", type=float, default=0.05)
    parser.add_argument(
        "--history-policy", choices=("capture_reset", "continuous"), default="continuous"
    )
    args = parser.parse_args()
    if not 0 < args.fpr < 1 or args.buffer < math.ceil(1 / args.fpr) - 1:
        parser.error("buffer cannot support the requested calibration FPR")
    run(args)


if __name__ == "__main__":
    main()
