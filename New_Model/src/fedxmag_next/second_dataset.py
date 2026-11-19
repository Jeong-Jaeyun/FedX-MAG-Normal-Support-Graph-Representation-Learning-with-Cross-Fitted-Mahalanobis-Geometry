"""Second-dataset architectural check on CIC-IDS2017 actual full windows.

Learn only from Monday normal train/selection roles; query Thursday/Friday.
Use only strictly earlier actual normal windows for rolling calibration. This
tests immediate oracle normal verification, not background-record filtering.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .capture_experiment import GROUPS
from .contracts import load_windows
from .contrastive_energy import DependencyEnergy, train
from .detector import conformal_threshold
from .experiments import aggregate_rows, knn_scores, sha256, write_json
from .geometry import ConditionalMahalanobis


def evaluate_rolling(scores: dict[str, np.ndarray], windows: list, buffer: int = 20) -> dict:
    order = sorted(
        range(len(windows)),
        key=lambda i: (windows[i].metadata["session_key"], windows[i].timestamp()),
    )
    result = {}
    for method, values in scores.items():
        history = defaultdict(lambda: deque(maxlen=buffer))
        flags, labels, by_type, kept, by_capture = [], [], defaultdict(list), [], defaultdict(list)
        for index in order:
            window = windows[index]
            capture = window.metadata["session_key"]
            threshold = conformal_threshold(history[capture], 0.05)
            if len(history[capture]) >= buffer and threshold is not None:
                alert = bool(values[index] > threshold)
                flags.append(alert)
                labels.append(window.label)
                kept.append(index)
                if window.label == 0:
                    by_capture[capture].append(alert)
                else:
                    names = window.metadata.get("attack_types", []) or ["unknown"]
                    for name in names:
                        by_type[name].append(alert)
            # Score, then accept externally verified normal feedback. Query
            # energy and alert status do not decide normal membership.
            if window.label == 0:
                history[capture].append(float(values[index]))
        labels, flags = np.asarray(labels), np.asarray(flags)
        result[method] = {
            "tpr": float(flags[labels == 1].mean()),
            "fpr": float(flags[labels == 0].mean()),
            "auroc": float(roc_auc_score(labels, values[kept])),
            "evaluated_windows": len(kept),
            "total_windows": len(windows),
            "normal_count": int(np.sum(labels == 0)),
            "attack_count": int(np.sum(labels == 1)),
            "per_capture_fpr": {
                name: float(np.mean(alerts)) for name, alerts in by_capture.items()
            },
            "per_type_tpr": {
                name: {"windows": len(alerts), "tpr": float(np.mean(alerts))}
                for name, alerts in by_type.items()
            },
        }
    return result


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    folder = (
        "cic_ids2017_validation_20261101"
        if args.dataset == "cic"
        else "ton_iot_validation_20261027"
    )
    view = root / "output" / folder / "views"
    target_name = "window" if args.dataset == "cic" else "attack"
    source_path, target_path = view / "source_graphs.pt", view / f"{target_name}_graphs.pt"
    source, target = load_windows(source_path), load_windows(target_path)
    source_flows = torch.load(view / "source_flows.pt", weights_only=True)
    target_flows = torch.load(view / f"{target_name}_flows.pt", weights_only=True)
    extra_inputs = []
    if args.dataset == "ton":
        background = load_windows(view / "background_graphs.pt")
        background_flows = torch.load(view / "background_flows.pt", weights_only=True)
        # Natural-normal backgrounds contain every actual row in that window.
        # Filtered label-0 fragments of attack windows are excluded entirely.
        for window, flow in zip(background, background_flows, strict=True):
            if window.metadata.get("natural_normal") is True:
                target.append(window)
                target_flows.append(flow)
        extra_inputs = ["background_graphs.pt", "background_flows.pt"]
    if len(source) != len(source_flows) or len(target) != len(target_flows):
        raise ValueError("CIC flow/graph window count mismatch")
    indices = {
        role: [i for i, window in enumerate(source) if window.metadata["role"] == role]
        for role in ("train", "selection")
    }
    if any(source[i].label != 0 for members in indices.values() for i in members):
        raise ValueError("Non-normal Monday fitting window")
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": args.dataset,
            "status": "second_dataset_development",
            "inputs": {
                name: sha256(view / name)
                for name in [
                    "source_graphs.pt",
                    "source_flows.pt",
                    f"{target_name}_graphs.pt",
                    f"{target_name}_flows.pt",
                    *extra_inputs,
                ]
            },
            "training": "source normal train; selection-normal synthetic BCE",
            "evaluation": "actual full target windows; no filtered background samples",
            "calibration": "20 most recent strictly earlier externally verified normal windows",
            "counts": {
                "train": len(indices["train"]),
                "selection": len(indices["selection"]),
                "target": len(target),
            },
        },
    )
    started = time.perf_counter()
    reference = torch.cat([source_flows[i] for i in indices["train"]]).float()
    selection = torch.cat([source_flows[i] for i in indices["selection"]]).float()
    mean, scale = reference.mean(dim=0), reference.std(dim=0, unbiased=True)
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    reference, selection = (reference - mean) / scale, (selection - mean) / scale
    query = (torch.cat(target_flows).float() - mean) / scale
    # Fixed hash-free deterministic subsample bounds CPU cost equally for both
    # reference-based methods; no evaluation label selects support rows.
    if len(reference) > 20000:
        order = torch.randperm(len(reference), generator=torch.Generator().manual_seed(71))[:20000]
        knn_reference = reference[order]
    else:
        knn_reference = reference
    print({"phase": "cic_fit", "source_rows": len(reference), "query_rows": len(query)}, flush=True)
    baseline = aggregate_rows(knn_scores(knn_reference, query), target_flows)
    selection_knn = aggregate_rows(
        knn_scores(knn_reference, selection), [source_flows[i] for i in indices["selection"]]
    )
    baseline = baseline / max(float(np.quantile(selection_knn, 0.95)), 1e-9)
    torch.manual_seed(args.seed)
    model = DependencyEnergy(reference.shape[1])
    history = train(model, reference, selection, GROUPS, seed=args.seed, epochs=args.epochs)
    with torch.no_grad():
        neural = aggregate_rows(model(query), target_flows)
        selected = aggregate_rows(model(selection), [source_flows[i] for i in indices["selection"]])
    median, q95 = np.quantile(selected, [0.5, 0.95])
    neural = ((neural - median) / max(q95 - median, 1e-6) + 1.0) / 2.0
    scores = {
        "flow_knn": baseline,
        "dependency_energy": neural,
        "flow_knn_plus_dependency": np.maximum(baseline, neural),
        "record_count": np.asarray([np.log1p(w.metadata["records"]) for w in target]),
    }
    with torch.no_grad():
        train_features = model.layers[:-1](reference)
        selection_features = model.layers[:-1](selection)
        target_features = model.layers[:-1](query)
        feature_mean = train_features.mean(dim=0)
        feature_scale = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
        support = ConditionalMahalanobis.fit(
            (train_features - feature_mean) / feature_scale, max_components=1
        )
        selected_energy = aggregate_rows(
            support.score((selection_features - feature_mean) / feature_scale)[0],
            [source_flows[i] for i in indices["selection"]],
        )
        energy = aggregate_rows(
            support.score((target_features - feature_mean) / feature_scale)[0], target_flows
        )
    median, q95 = np.quantile(selected_energy, [0.5, 0.95])
    energy = ((energy - median) / max(q95 - median, 1e-6) + 1) / 2
    scores["deep_mahalanobis"] = energy
    scores["flow_knn_plus_deep_mahalanobis"] = np.maximum(baseline, energy)
    report = {
        "status": "second_dataset_development",
        "methods": evaluate_rolling(scores, target),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "report.json", report)
    torch.save(
        {"model": model.state_dict(), "mean": mean, "scale": scale, "history": history},
        output / "model.pt",
    )
    torch.save(
        {name: torch.from_numpy(values) for name, values in scores.items()}, output / "scores.pt"
    )
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dataset", choices=("cic", "ton"), default="cic")
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
