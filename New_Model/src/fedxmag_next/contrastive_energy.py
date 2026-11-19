"""Normal-only noise-contrastive dependency energy for flow/entity rows.

Synthetic negatives permute whole valid feature groups from other normal rows.
No real attack example or evaluation label participates in learning.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from .experiments import aggregate_rows, fuse, load_roles, raw_rows, sha256, summarize, write_json
from .flow_experiment import signed_scale


class DependencyEnergy(nn.Module):
    def __init__(self, dimension: int, hidden: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values).squeeze(-1)


def corrupt(values: Tensor, groups: list[list[int]], generator: torch.Generator) -> Tensor:
    changed = values.clone()
    for group in groups:
        order = torch.randperm(len(values), generator=generator)
        chosen = torch.nonzero(torch.rand(len(values), generator=generator) < 0.5).flatten()
        if len(chosen):
            changed[chosen[:, None], torch.tensor(group)[None, :]] = values[order[chosen]][:, group]
    return changed


def train(
    model: DependencyEnergy,
    reference: Tensor,
    selection: Tensor,
    groups: list[list[int]],
    *,
    seed: int,
    epochs: int,
    include_initial: bool = False,
) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_function = nn.BCEWithLogitsLoss()
    history = []
    best_value, best_state = float("inf"), None
    validation_generator = torch.Generator().manual_seed(7189)
    validation_negative = corrupt(selection, groups, validation_generator)
    if include_initial:
        model.eval()
        with torch.no_grad():
            best_value = float(
                loss_function(
                    model(torch.cat((selection, validation_negative))),
                    torch.cat((torch.zeros(len(selection)), torch.ones(len(selection)))),
                )
            )
        best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": 0, "selection_synthetic_bce": best_value})
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(reference), generator=generator)
        total = 0.0
        batches = 0
        for indices in order.split(256):
            positive = reference[indices]
            negative = corrupt(positive, groups, generator)
            values = torch.cat((positive, negative))
            labels = torch.cat((torch.zeros(len(positive)), torch.ones(len(negative))))
            loss = loss_function(model(values), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        model.eval()
        with torch.no_grad():
            validation = float(
                loss_function(
                    model(torch.cat((selection, validation_negative))),
                    torch.cat((torch.zeros(len(selection)), torch.ones(len(selection)))),
                )
            )
        history.append(
            {"epoch": epoch, "train_bce": total / batches, "selection_synthetic_bce": validation}
        )
        if validation < best_value:
            best_value = validation
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(history[-1], flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    root, destination = Path(args.root).resolve(), Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    windows, paths = load_roles(root)
    cache_path = root / "output/root_cause_20261026/flow_context_cache.pt"
    cache = torch.load(cache_path, weights_only=True)
    archived = torch.load(root / "output/fusion_detector_v2/replay_scores.pt", weights_only=True)
    baseline = {role: values.numpy() for role, values in archived["fusion"].items()}
    write_json(
        destination / "registration.json",
        {
            "args": vars(args),
            "status": "iterative_development_on_reused_datasets",
            "inputs": {role: sha256(path) for role, path in paths.items()},
            "flow_cache_sha256": sha256(cache_path),
            "selection": "normal-vs-synthetic BCE on selection-normal only",
            "hypothesis": (
                "Learn valid-group dependency violations using normal-only synthetic negatives"
            ),
            "baseline": "exact archived raw-node+flow kNN scores, same windows and calibration",
        },
    )
    start = time.perf_counter()
    methods = {"archived_raw_node_flow_knn": baseline}
    new_channels = []
    for kind in ("flow", "node"):
        if kind == "flow":
            rows = cache["features"]
            groups = [[0], [1, 2, 3], [4, 5, 6], list(range(7, 12)), [12], [13], [14], [15], [16]]
        else:
            rows = {role: raw_rows(items, "node") for role, items in windows.items()}
            groups = [
                [*block, *(i + 14 for i in block)]
                for block in (range(0, 4), range(4, 8), range(8, 11), range(11, 14))
            ]
        reference = torch.cat(rows["train"]).float()
        mean = reference.mean(dim=0)
        scale = reference.std(dim=0, unbiased=False).clamp_min(1e-6)
        matrices = {role: (torch.cat(value).float() - mean) / scale for role, value in rows.items()}
        torch.manual_seed(args.seed)
        model = DependencyEnergy(reference.shape[1])
        history = train(
            model,
            matrices["train"],
            matrices["selection"],
            groups,
            seed=args.seed,
            epochs=args.epochs,
        )
        with torch.no_grad():
            scores = {
                role: aggregate_rows(model(value), rows[role])
                for role, value in matrices.items()
                if role != "train"
            }
        channel = signed_scale(scores)
        new_channels.append(channel)
        methods[f"{kind}_dependency_energy"] = channel
        methods[f"raw_plus_{kind}_dependency"] = fuse([baseline, channel])
        torch.save(
            {
                "model": model.state_dict(),
                "dimension": reference.shape[1],
                "mean": mean,
                "scale": scale,
                "groups": groups,
                "history": history,
            },
            destination / f"{kind}_model.pt",
        )
    methods["node_flow_dependency"] = fuse(new_channels)
    methods["raw_plus_node_flow_dependency"] = fuse([baseline, *new_channels])
    report = {"status": "iterative_development", "methods": {}}
    for name, scores in methods.items():
        result = summarize(scores, windows)
        report["methods"][name] = result
        print(
            json.dumps(
                {
                    "method": name,
                    "metrics": {
                        band: {key: entry[key] for key in ("auroc", "tpr", "fpr")}
                        for band, entry in result.items()
                    },
                }
            ),
            flush=True,
        )
    report["elapsed_seconds"] = time.perf_counter() - start
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
