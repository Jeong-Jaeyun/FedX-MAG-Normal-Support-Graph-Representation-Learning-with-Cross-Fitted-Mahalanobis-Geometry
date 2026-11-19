"""Capture-disjoint development stress test of normal-only dependency energy.

All scaling, normal references, and neural weights are refitted per fold. A
separate source-calibration capture fold supplies every operational threshold.
The archived GNN or flow models are never reused across this isolation boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .contrastive_energy import DependencyEnergy, train
from .experiments import (
    aggregate_rows,
    fuse,
    knn_scores,
    load_roles,
    metrics,
    sha256,
    write_json,
)
from .flow_experiment import signed_scale

GROUPS = [[0], [1, 2, 3], [4, 5, 6], list(range(7, 12)), [12], [13], [14], [15], [16]]


def node_refit(windows: list, training: list[int], archived_scaler: dict) -> list[torch.Tensor]:
    """Undo archived scaling, then graph-balance the new source-only scaler."""

    available = [w.node_available for w in windows]
    raw = [
        w.node_features.double() * archived_scaler["node_scale"].double()
        + archived_scaler["node_mean"].double()
        for w in windows
    ]
    counts = [available[index].sum(dim=0).clamp_min(1) for index in training]
    observed = torch.stack([available[index].any(dim=0) for index in training])
    observed_count = observed.sum(dim=0).clamp_min(1)
    mean = (
        torch.stack(
            [
                (raw[index] * available[index]).sum(dim=0) / count
                for index, count in zip(training, counts, strict=True)
            ]
        )
        * observed
    ).sum(dim=0) / observed_count
    variance = (
        torch.stack(
            [
                ((raw[index] - mean) * available[index]).square().sum(dim=0) / count
                for index, count in zip(training, counts, strict=True)
            ]
        )
        * observed
    ).sum(dim=0) / observed_count
    scale = variance.sqrt().clamp_min(1e-6)
    return [
        torch.cat((((value - mean) / scale) * mask, mask.double()), dim=1)
        for value, mask in zip(raw, available, strict=True)
    ]


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    root, destination = Path(args.root).resolve(), Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    roles, paths = load_roles(root)
    flow_path = root / "output/root_cause_20261026/flow_context_cache.pt"
    cache = torch.load(flow_path, weights_only=True)
    windows = [window for role in roles for window in roles[role]]
    flows = [row for role in roles for row in cache["features"][role]]
    sessions = [w.metadata["session_key"] for w in windows]
    captures = sorted(set(sessions), key=lambda key: hashlib.sha256(key.encode()).hexdigest())
    assignment = {capture: index % 5 for index, capture in enumerate(captures)}
    folds = np.asarray([assignment[capture] for capture in sessions])
    labels = np.asarray([w.label for w in windows])
    keys = [f"{w.metadata['session_key']}:{w.timestamp()}" for w in windows]
    scaler = torch.load(paths["train"].parent / "standardizer.pt", weights_only=True)
    write_json(
        destination / "registration.json",
        {
            "status": "capture_disjoint_development_stress",
            "args": vars(args),
            "inputs": {role: sha256(path) for role, path in paths.items()},
            "flow_cache_sha256": sha256(flow_path),
            "assignment": assignment,
            "isolation": (
                "Evaluation fold and next-fold calibration are excluded "
                "from all model/scaler fitting"
            ),
            "selection": (
                "hash-ordered 20% of source-normal windows; synthetic-normal validation only"
            ),
            "all_folds_reported": True,
        },
    )
    report = {"status": "capture_disjoint_development_stress", "folds": {}}
    score_archive = {}
    start = time.perf_counter()
    for fold in args.folds:
        source = np.flatnonzero(
            (folds != fold) & (folds != ((fold + 1) % 5)) & (labels == 0)
        ).tolist()
        source.sort(key=lambda i: hashlib.sha256(keys[i].encode()).hexdigest())
        split = int(0.8 * len(source))
        indices = {
            "train": source[:split],
            "selection": source[split:],
            "calibration": np.flatnonzero((folds == (fold + 1) % 5) & (labels == 0)).tolist(),
            "normal": np.flatnonzero((folds == fold) & (labels == 0)).tolist(),
            "attack": np.flatnonzero((folds == fold) & (labels == 1)).tolist(),
        }
        source_captures = {sessions[index] for index in indices["train"]}
        cal_captures = {sessions[index] for index in indices["calibration"]}
        eval_captures = {sessions[index] for index in indices["normal"] + indices["attack"]}
        assert not source_captures & cal_captures
        assert not source_captures & eval_captures
        assert not cal_captures & eval_captures
        print(
            json.dumps(
                {
                    "phase": "capture_fold",
                    "fold": fold,
                    "counts": {r: len(i) for r, i in indices.items()},
                }
            ),
            flush=True,
        )
        node_rows = node_refit(windows, indices["train"], scaler)
        flow_train = torch.cat([flows[index] for index in indices["train"]]).float()
        mean, scale = flow_train.mean(dim=0), flow_train.std(dim=0, unbiased=True)
        # Keep all dimensions for the DNN, with explicit finite scales. kNN
        # includes constant coordinates to detect genuinely unseen activity.
        scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
        matrices = {
            role: (torch.cat([flows[index] for index in members]).float() - mean) / scale
            for role, members in indices.items()
        }
        channels = []
        for rows in (node_rows, [(flow.float() - mean) / scale for flow in flows]):
            reference = torch.cat([rows[index] for index in indices["train"]])
            channel = {}
            for role, members in indices.items():
                if role == "train":
                    continue
                query_rows = [rows[index] for index in members]
                channel[role] = aggregate_rows(
                    knn_scores(reference, torch.cat(query_rows)), query_rows
                )
            channels.append(channel)
        baseline = fuse(channels)
        torch.manual_seed(args.seed)
        model = DependencyEnergy(flow_train.shape[1])
        history = train(
            model,
            matrices["train"],
            matrices["selection"],
            GROUPS,
            seed=args.seed,
            epochs=args.epochs,
        )
        with torch.no_grad():
            neural_scores = {
                role: aggregate_rows(model(values), [flows[index] for index in indices[role]])
                for role, values in matrices.items()
                if role != "train"
            }
        neural = signed_scale(neural_scores)
        methods = {
            "raw_node_flow_knn": baseline,
            "flow_dependency": neural,
            "raw_plus_flow_dependency": fuse([baseline, neural]),
        }
        results = {
            name: metrics(scores["calibration"], scores["normal"], scores["attack"])
            for name, scores in methods.items()
        }
        for name, values in methods.items():
            threshold = results[name]["threshold"]
            normal_sessions = np.asarray([sessions[index] for index in indices["normal"]])
            results[name]["per_capture_fpr"] = {
                capture: {
                    "normal_count": int(np.sum(normal_sessions == capture)),
                    "fpr": float(np.mean(values["normal"][normal_sessions == capture] > threshold)),
                }
                for capture in sorted(set(normal_sessions))
            }
        entry = {
            "counts": {role: len(items) for role, items in indices.items()},
            "source_captures": sorted(source_captures),
            "calibration_captures": sorted(cal_captures),
            "evaluation_captures": sorted(eval_captures),
            "methods": results,
        }
        report["folds"][str(fold)] = entry
        write_json(destination / f"fold_{fold}.json", entry)
        print(
            json.dumps(
                {
                    "fold": fold,
                    "methods": {
                        name: {k: m[k] for k in ("auroc", "tpr", "fpr")}
                        for name, m in results.items()
                    },
                }
            ),
            flush=True,
        )
        torch.save(
            {"model": model.state_dict(), "mean": mean, "scale": scale, "history": history},
            destination / f"fold_{fold}_model.pt",
        )
        score_archive[str(fold)] = {
            "indices": indices,
            "methods": {
                name: {role: torch.from_numpy(value) for role, value in scores.items()}
                for name, scores in methods.items()
            },
        }
    report["elapsed_seconds"] = time.perf_counter() - start
    write_json(destination / "report.json", report)
    torch.save(score_archive, destination / "scores.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
