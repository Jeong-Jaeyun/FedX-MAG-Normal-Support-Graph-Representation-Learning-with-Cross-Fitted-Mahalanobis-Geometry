"""Fixed mean/median neural ensemble controls across both datasets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from .contracts import load_windows
from .detector import conformal_threshold
from .experiments import fuse, load_roles, summarize, write_json
from .second_dataset import evaluate_rolling


def rolling_trace(values: np.ndarray, windows: list, size: int = 20) -> np.ndarray:
    trace = np.full(len(windows), np.nan)
    order = sorted(
        range(len(windows)),
        key=lambda i: (windows[i].metadata["session_key"], windows[i].timestamp()),
    )
    history = defaultdict(lambda: deque(maxlen=size))
    for index in order:
        window = windows[index]
        capture = window.metadata["session_key"]
        if len(history[capture]) >= size:
            threshold = conformal_threshold(history[capture], 0.05)
            trace[index] = float(values[index] > threshold)
        if window.label == 0:
            history[capture].append(float(values[index]))
    return trace


def block_interval(candidate: np.ndarray, baseline: np.ndarray, windows: list) -> dict:
    valid = np.isfinite(candidate) & np.isfinite(baseline)
    labels = np.asarray([window.label for window in windows])
    # UTC hour and capture form the sampling block; their absolute time offset
    # does not affect grouping or any prediction.
    blocks = [
        (window.metadata["session_key"], window.timestamp() // (3600 * 10**9)) for window in windows
    ]
    groups = sorted({block for index, block in enumerate(blocks) if valid[index]})
    lookup = {block: index for index, block in enumerate(groups)}
    block_index = np.asarray([lookup.get(block, -1) for block in blocks])
    generator = np.random.default_rng(90210)
    differences = {"tpr": [], "fpr": []}
    for _ in range(2000):
        counts = np.bincount(
            generator.integers(len(groups), size=len(groups)), minlength=len(groups)
        )
        weights = np.asarray([counts[index] if index >= 0 else 0 for index in block_index])
        for key, label in (("tpr", 1), ("fpr", 0)):
            chosen = valid & (labels == label)
            if weights[chosen].sum():
                differences[key].append(
                    float(np.average(candidate[chosen] - baseline[chosen], weights=weights[chosen]))
                )
    return {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "hour_blocks": len(groups),
        }
        for key, values in differences.items()
    }


def run(args: argparse.Namespace) -> None:
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    runs = root / "New_Model/runs"
    seeds = [17, 29, 41]
    report = {"status": "fixed_three_seed_ensemble_development", "xiiotid": {}, "cic": {}}
    xwindows, _ = load_roles(root)
    source = [
        torch.load(runs / f"round03_dependency_seed{s}/scores.pt", weights_only=True) for s in seeds
    ]
    baseline = {
        role: value.numpy() for role, value in source[0]["archived_raw_node_flow_knn"].items()
    }
    channels = [
        {role: value.numpy() for role, value in scores["flow_dependency_energy"].items()}
        for scores in source
    ]
    for name, reducer in (("mean", np.mean), ("median", np.median)):
        channel = {
            role: reducer([values[role] for values in channels], axis=0) for role in channels[0]
        }
        combined = fuse([baseline, channel])
        report["xiiotid"][name] = summarize(combined, xwindows)
    windows = load_windows(root / "output/cic_ids2017_validation_20261101/views/window_graphs.pt")
    scores = [
        torch.load(runs / f"round06_cic_dependency_seed{s}/scores.pt", weights_only=True)
        for s in seeds
    ]
    baseline_cic = scores[0]["flow_knn"].numpy()
    methods = {"flow_knn": baseline_cic, "record_count": scores[0]["record_count"].numpy()}
    for name, reducer in (("mean", np.mean), ("median", np.median)):
        neural = reducer([s["dependency_energy"].numpy() for s in scores], axis=0)
        methods[f"{name}_neural"] = neural
        methods[f"raw_max_{name}_neural"] = np.maximum(baseline_cic, neural)
        methods[f"raw_sum_{name}_neural"] = baseline_cic + neural
    report["cic"] = evaluate_rolling(methods, windows)
    feature_path = runs / "round07_deep_mahalanobis_cic/scores.pt"
    xfeature_path = runs / "round07_deep_mahalanobis_xiiotid/scores.pt"
    if feature_path.exists() and xfeature_path.exists():
        feature = torch.load(feature_path, weights_only=True)
        xfeature = torch.load(xfeature_path, weights_only=True)
        report["xiiotid_deep_feature_ensembles"] = {}
        for modes in (1, 8):
            for name, reducer in (("mean", np.mean), ("median", np.median)):
                neural = reducer(
                    [
                        feature[f"deep_mahalanobis_k{modes}_seed{seed}"]["target"].numpy()
                        for seed in seeds
                    ],
                    axis=0,
                )
                methods[f"deep_m{modes}_{name}"] = neural
                methods[f"raw_max_deep_m{modes}_{name}"] = np.maximum(baseline_cic, neural)
                xchannel = {
                    role: reducer(
                        [
                            xfeature[f"deep_mahalanobis_k{modes}_seed{seed}"][role].numpy()
                            for seed in seeds
                        ],
                        axis=0,
                    )
                    for role in baseline
                }
                report["xiiotid_deep_feature_ensembles"][f"m{modes}_{name}"] = summarize(
                    fuse([baseline, xchannel]), xwindows
                )
        report["cic"] = evaluate_rolling(methods, windows)
    baseline_trace = rolling_trace(baseline_cic, windows)
    report["cic_hour_bootstrap_differences"] = {
        name: block_interval(rolling_trace(values, windows), baseline_trace, windows)
        for name, values in methods.items()
        if name not in {"flow_knn", "record_count"}
    }
    write_json(output / "report.json", report)
    torch.save(
        {name: torch.from_numpy(values) for name, values in methods.items()}, output / "scores.pt"
    )
    print(
        json.dumps(
            {
                name: {key: result[key] for key in ("auroc", "tpr", "fpr")}
                for name, result in report["cic"].items()
            }
        ),
        flush=True,
    )
    print(json.dumps(report["cic_hour_bootstrap_differences"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
