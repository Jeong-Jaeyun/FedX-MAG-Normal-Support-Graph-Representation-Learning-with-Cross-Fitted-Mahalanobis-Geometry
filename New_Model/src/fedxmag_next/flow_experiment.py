"""Round 2: information-preserving deep energy and raw support controls."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .experiments import (
    aggregate_rows,
    channel_scores,
    fuse,
    load_roles,
    raw_rows,
    sha256,
    summarize,
    write_json,
)
from .flow_energy import FlowConfig, InvertibleEnergy, train_flow


def signed_scale(channel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    median, q95 = np.quantile(channel["selection"], [0.5, 0.95])
    scale = max(q95 - median, 1e-6)
    return {role: (values - median) / scale + 1.0 for role, values in channel.items()}


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    root, destination = Path(args.root).resolve(), Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    windows, paths = load_roles(root)
    cache_path = root / "output/root_cause_20261026/flow_context_cache.pt"
    flows = torch.load(cache_path, weights_only=True)["features"]
    # Restore the exact archived baseline for the main comparison.
    baseline_path = root / "output/fusion_detector_v2/replay_scores.pt"
    archived = torch.load(baseline_path, weights_only=True)
    print({"archived_keys": list(archived)}, flush=True)
    write_json(
        destination / "registration.json",
        {
            "status": "iterative_development",
            "args": vars(args),
            "inputs": {role: sha256(path) for role, path in paths.items()},
            "flow_cache_sha256": sha256(cache_path),
            "baseline_sha256": sha256(baseline_path),
            "model_selection": "selection-normal NLL only; all score variants reported",
            "hypothesis": (
                "An invertible deep map retains anomaly evidence "
                "while modeling nonlinear normal support."
            ),
        },
    )
    started = time.perf_counter()
    channels = {}
    for kind, rows in (
        ("node", {role: raw_rows(items, "node") for role, items in windows.items()}),
        ("flow", flows),
    ):
        channels[kind] = channel_scores(rows, "knn")
    baseline = fuse(list(channels.values()))
    methods = {"raw_standardized_node_flow_knn": baseline}
    if "fusion" in archived:
        methods["archived_raw_node_flow_knn"] = {
            role: values.numpy() for role, values in archived["fusion"].items()
        }
    for kind in ("flow", "node"):
        rows = (
            flows
            if kind == "flow"
            else {role: raw_rows(items, "node") for role, items in windows.items()}
        )
        reference = torch.cat(rows["train"]).float()
        scale = reference.std(dim=0, unbiased=False)
        keep = scale > 1e-6
        mean, scale = reference.mean(dim=0)[keep], scale[keep]
        matrices = {
            role: (torch.cat(items).float()[:, keep] - mean) / scale for role, items in rows.items()
        }
        torch.manual_seed(args.seed)
        config = FlowConfig(int(keep.sum()), layers=6, hidden=64)
        model = InvertibleEnergy(config)
        history = train_flow(
            model, matrices["train"], matrices["selection"], epochs=args.epochs, seed=args.seed
        )
        with torch.no_grad():
            outputs = {role: model.scores(matrix) for role, matrix in matrices.items()}
        for scoring in ("energy", "nll"):
            scores = {
                role: aggregate_rows(values[scoring], rows[role])
                for role, values in outputs.items()
                if role != "train"
            }
            methods[f"{kind}_invertible_{scoring}"] = signed_scale(scores)
            methods[f"raw_plus_{kind}_invertible_{scoring}"] = fuse(
                [
                    *channels.values(),
                    signed_scale(scores),
                ]
            )
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(config),
                "mean": mean,
                "scale": scale,
                "keep": keep,
                "history": history,
            },
            destination / f"{kind}_flow.pt",
        )
    for scoring in ("energy", "nll"):
        methods[f"node_flow_invertible_{scoring}"] = fuse(
            [
                methods[f"node_invertible_{scoring}"],
                methods[f"flow_invertible_{scoring}"],
            ]
        )
    report = {"status": "iterative_development", "methods": {}}
    for name, scores in methods.items():
        report["methods"][name] = summarize(scores, windows)
        print(
            json.dumps(
                {
                    "method": name,
                    "metrics": {
                        band: {key: result[key] for key in ("auroc", "tpr", "fpr")}
                        for band, result in report["methods"][name].items()
                    },
                }
            ),
            flush=True,
        )
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(destination / "report.json", report)
    torch.save(
        {
            name: {role: torch.from_numpy(value) for role, value in scores.items()}
            for name, scores in methods.items()
        },
        destination / "scores.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
