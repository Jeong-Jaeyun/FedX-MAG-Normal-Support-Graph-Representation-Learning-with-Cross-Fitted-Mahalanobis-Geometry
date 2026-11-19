"""Deep Mahalanobis energy on contrastively trained dependency features.

This tests whether the representation learned to distinguish normal dependence
is more useful for normal-support scoring than masked reconstruction features.
Existing neural weights are frozen, normal-reference statistics are fitted anew.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .contracts import load_windows
from .contrastive_energy import DependencyEnergy
from .experiments import aggregate_rows, fuse, load_roles, summarize, write_json
from .flow_experiment import signed_scale
from .geometry import ConditionalMahalanobis
from .second_dataset import evaluate_rolling


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.dataset == "xiiotid":
        windows, _ = load_roles(root)
        rows = torch.load(
            root / "output/root_cause_20261026/flow_context_cache.pt", weights_only=True
        )["features"]
        baseline = {
            role: value.numpy()
            for role, value in torch.load(
                root / "output/fusion_detector_v2/replay_scores.pt", weights_only=True
            )["fusion"].items()
        }
        targets = None
    else:
        view = root / "output/cic_ids2017_validation_20261101/views"
        source = load_windows(view / "source_graphs.pt")
        targets = load_windows(view / "window_graphs.pt")
        source_rows = torch.load(view / "source_flows.pt", weights_only=True)
        rows = {
            role: [
                row
                for row, window in zip(source_rows, source, strict=True)
                if window.metadata["role"] == role
            ]
            for role in ("train", "selection")
        }
        rows["target"] = torch.load(view / "window_flows.pt", weights_only=True)
        baseline = {
            "target": torch.load(
                root / "New_Model/runs/round06_cic_dependency_seed17/scores.pt", weights_only=True
            )["flow_knn"].numpy()
        }
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "status": "deep_feature_energy_development",
            "weights": "frozen normal-only dependency classifier",
            "support": "normal train only, 1 and 8 shrinkage-Mahalanobis modes",
            "all_seed_and_geometry_variants_reported": True,
        },
    )
    all_scores = {}
    for seed in args.seeds:
        if args.dataset == "xiiotid":
            checkpoint = root / f"New_Model/runs/round03_dependency_seed{seed}/flow_model.pt"
        else:
            checkpoint = root / f"New_Model/runs/round06_cic_dependency_seed{seed}/model.pt"
        state = torch.load(checkpoint, weights_only=True)
        model = DependencyEnergy(len(state["mean"]))
        model.load_state_dict(state["model"])
        model.eval()
        with torch.no_grad():
            embeddings = {
                role: model.layers[:-1]((torch.cat(items).float() - state["mean"]) / state["scale"])
                for role, items in rows.items()
            }
        reference = embeddings["train"]
        scale = reference.std(dim=0, unbiased=False)
        keep = scale > 1e-6
        mean, scale = reference.mean(dim=0)[keep], scale[keep]
        embeddings = {role: (value[:, keep] - mean) / scale for role, value in embeddings.items()}
        for modes in (1, 8):
            support = ConditionalMahalanobis.fit(
                embeddings["train"], max_components=modes, min_component_size=64, seed=17
            )
            energies = {
                role: aggregate_rows(support.score(values)[0], rows[role])
                for role, values in embeddings.items()
                if role != "train"
            }
            scaled = signed_scale(energies)
            name = f"deep_mahalanobis_k{modes}_seed{seed}"
            if args.dataset == "xiiotid":
                all_scores[name] = scaled
                all_scores[f"raw_plus_{name}"] = fuse([baseline, scaled])
            else:
                neural = scaled["target"] / 2
                all_scores[name] = {"target": neural}
                all_scores[f"raw_plus_{name}"] = {"target": np.maximum(baseline["target"], neural)}
    results = {}
    for name, scores in all_scores.items():
        if args.dataset == "xiiotid":
            result = summarize(scores, windows)
            brief = {
                band: {key: entry[key] for key in ("auroc", "tpr", "fpr")}
                for band, entry in result.items()
            }
        else:
            result = evaluate_rolling({name: scores["target"]}, targets)[name]
            brief = {key: result[key] for key in ("auroc", "tpr", "fpr")}
        results[name] = result
        print(json.dumps({"method": name, "metrics": brief}), flush=True)
    write_json(
        output / "report.json", {"status": "deep_feature_energy_development", "methods": results}
    )
    torch.save(
        {
            name: {role: torch.from_numpy(value) for role, value in scores.items()}
            for name, scores in all_scores.items()
        },
        output / "scores.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", choices=("xiiotid", "cic"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 41])
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
