"""Small, explicit M0 loops that preserve per-family denominators."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from statistics import mean, pstdev
from typing import Any

import torch
from torch import nn

from fedxmag.losses import FAMILY_KEYS, m0_loss
from fedxmag.masking import FamilyMasker, MaskedGraphBatch
from fedxmag.models.m0 import M0Output
from fedxmag.schema import DEFAULT_FEATURE_GROUPS, FeatureGroups, GraphSample, collate_graphs

SELECTION_METRICS = {
    "validation_mask_mean_macro_family_mse": "macro_family_mse_mean",
    "validation_mask_mean_worst_family_mse": "worst_family_mse_mean",
}


def _batches(
    samples: Sequence[GraphSample],
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator,
) -> Iterator[list[GraphSample]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if shuffle:
        order = torch.randperm(len(samples), generator=generator).tolist()
    else:
        order = list(range(len(samples)))
    for start in range(0, len(order), batch_size):
        yield [samples[index] for index in order[start : start + batch_size]]


def _degree_bin_masks(degrees: torch.Tensor) -> tuple[tuple[str, torch.Tensor], ...]:
    return (
        ("0-1", degrees <= 1),
        ("2-3", (degrees >= 2) & (degrees <= 3)),
        ("4+", degrees >= 4),
    )


def _accumulate_degree_strata(
    output: M0Output,
    masked: MaskedGraphBatch,
    error_sums: dict[str, float],
    counts: dict[str, int],
    graph_counts: dict[str, int],
    groups: FeatureGroups = DEFAULT_FEATURE_GROUPS,
) -> None:
    batch = masked.batch
    source, destination = batch.edge_index
    out_degree = torch.bincount(source, minlength=batch.node_features.shape[0])
    in_degree = torch.bincount(destination, minlength=batch.node_features.shape[0])

    def collect(
        key_prefix: str,
        prediction: torch.Tensor,
        target: torch.Tensor,
        targeted: torch.Tensor,
        graph_index: torch.Tensor,
        indices: list[int],
        degrees: torch.Tensor,
    ) -> None:
        squared_error = (prediction[:, indices] - target[:, indices]).square()
        selected = targeted[:, indices]
        for degree_name, degree_mask in _degree_bin_masks(degrees):
            key = f"{key_prefix}|degree={degree_name}"
            for graph in range(batch.num_graphs):
                rows = (graph_index == graph) & degree_mask
                if not bool(rows.any()):
                    continue
                scalar_mask = selected[rows]
                scalar_count = int(scalar_mask.sum())
                if not scalar_count:
                    continue
                error_sums[key] += float(squared_error[rows].masked_select(scalar_mask).mean())
                counts[key] += scalar_count
                graph_counts[key] += 1

    for family in ("timing", "distribution"):
        for direction, degrees in (("out", out_degree), ("in", in_degree)):
            collect(
                f"node_{family}_{direction}",
                output.node_prediction,
                masked.node_targets,
                masked.node_targeted,
                batch.node_graph_index,
                list(groups.directional_node_indices(family, direction)),
                degrees,
            )
        edge_degree = torch.maximum(out_degree[source], in_degree[destination])
        collect(
            f"edge_{family}",
            output.edge_prediction,
            masked.edge_targets,
            masked.edge_targeted,
            batch.edge_graph_index,
            list(groups.indices("edge", family)),
            edge_degree,
        )


@torch.no_grad()
def evaluate_m0(
    model: nn.Module,
    samples: Sequence[GraphSample],
    masker: FamilyMasker,
    batch_size: int = 32,
    seed: int = 10_007,
    device: str | torch.device = "cpu",
    loss_weights: Mapping[str, float] | None = None,
    groups: FeatureGroups = DEFAULT_FEATURE_GROUPS,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot evaluate an empty sample collection")
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    graph_error_sums = {key: 0.0 for key in FAMILY_KEYS}
    counts = {key: 0 for key in FAMILY_KEYS}
    graph_counts = {key: 0 for key in FAMILY_KEYS}
    stratum_error_sums: dict[str, float] = defaultdict(float)
    stratum_counts: dict[str, int] = defaultdict(int)
    stratum_graph_counts: dict[str, int] = defaultdict(int)

    for sample_batch in _batches(samples, batch_size, False, generator):
        batch = collate_graphs(sample_batch, groups).to(device)
        masked = masker(batch, generator=generator)
        output = model(masked)
        loss = m0_loss(output, masked, weights=loss_weights, groups=groups)
        for key in FAMILY_KEYS:
            graph_error_sums[key] += (
                float(loss.components[key]) * loss.graph_counts[key]
            )
            counts[key] += loss.counts[key]
            graph_counts[key] += loss.graph_counts[key]
        _accumulate_degree_strata(
            output,
            masked,
            stratum_error_sums,
            stratum_counts,
            stratum_graph_counts,
            groups,
        )

    metrics: dict[str, float] = {}
    active: list[float] = []
    for key in FAMILY_KEYS:
        value = (
            graph_error_sums[key] / graph_counts[key]
            if graph_counts[key]
            else float("nan")
        )
        metrics[f"{key}_mse"] = value
        metrics[f"{key}_count"] = float(counts[key])
        metrics[f"{key}_graph_count"] = float(graph_counts[key])
        if graph_counts[key]:
            active.append(value)
    metrics["macro_family_mse"] = sum(active) / len(active) if active else float("nan")
    metrics["degree_strata"] = {
        key: {
            "mse": stratum_error_sums[key] / stratum_graph_counts[key],
            "scalar_count": float(stratum_counts[key]),
            "graph_count": float(stratum_graph_counts[key]),
        }
        for key in sorted(stratum_graph_counts)
        if stratum_graph_counts[key]
    }
    return metrics


@torch.no_grad()
def evaluate_m0_repeated(
    model: nn.Module,
    samples: Sequence[GraphSample],
    masker: FamilyMasker,
    mask_seeds: Sequence[int],
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    loss_weights: Mapping[str, float] | None = None,
    groups: FeatureGroups = DEFAULT_FEATURE_GROUPS,
) -> dict[str, Any]:
    """Evaluate against several shared masks and expose mask-to-mask variability."""

    if not mask_seeds:
        raise ValueError("At least one evaluation mask seed is required")
    runs = [
        evaluate_m0(
            model,
            samples,
            masker,
            batch_size,
            seed,
            device,
            loss_weights,
            groups,
        )
        for seed in mask_seeds
    ]
    metric_keys = [key for key in runs[0] if key.endswith("_mse")]
    aggregate: dict[str, float] = {}
    for key in metric_keys:
        values = [run[key] for run in runs]
        finite = [value for value in values if torch.isfinite(torch.tensor(value))]
        aggregate[f"{key}_mean"] = mean(finite) if finite else float("nan")
        aggregate[f"{key}_std"] = pstdev(finite) if finite else float("nan")
    family_means = [aggregate[f"{key}_mse_mean"] for key in FAMILY_KEYS]
    aggregate["worst_family_mse_mean"] = max(family_means)
    stratum_keys = sorted(
        {
            key
            for run in runs
            for key in run.get("degree_strata", {})
        }
    )
    degree_strata: dict[str, dict[str, float]] = {}
    for key in stratum_keys:
        entries = [
            run["degree_strata"][key]
            for run in runs
            if key in run.get("degree_strata", {})
        ]
        values = [float(entry["mse"]) for entry in entries]
        degree_strata[key] = {
            "mse_mean": mean(values),
            "mse_std": pstdev(values),
            "scalar_count_mean": mean(float(entry["scalar_count"]) for entry in entries),
            "graph_count_mean": mean(float(entry["graph_count"]) for entry in entries),
        }
    return {
        "mask_seeds": list(mask_seeds),
        "aggregate": aggregate,
        "degree_strata": degree_strata,
        "per_mask": runs,
    }


def train_m0(
    model: nn.Module,
    train_samples: Sequence[GraphSample],
    validation_samples: Sequence[GraphSample],
    masker: FamilyMasker,
    epochs: int = 20,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 5.0,
    early_stopping_patience: int | None = None,
    validation_mask_seeds: Sequence[int] | None = None,
    selection_metric: str = "validation_mask_mean_macro_family_mse",
    loss_weights: Mapping[str, float] | None = None,
    seed: int = 17,
    device: str | torch.device = "cpu",
    groups: FeatureGroups = DEFAULT_FEATURE_GROUPS,
) -> list[dict[str, Any]]:
    if not train_samples or not validation_samples:
        raise ValueError("Both train and validation samples are required")
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    aggregate_selection_key = SELECTION_METRICS[selection_metric]
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    if validation_mask_seeds is None:
        validation_mask_seeds = (seed + 100_003, seed + 100_019, seed + 100_043)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        update_count = 0
        for sample_batch in _batches(train_samples, batch_size, True, generator):
            batch = collate_graphs(sample_batch, groups).to(device)
            masked = masker(batch, generator=generator)
            loss = m0_loss(
                model(masked), masked, weights=loss_weights, groups=groups
            )
            if not sum(loss.counts.values()):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            epoch_loss += float(loss.total.detach())
            update_count += 1

        validation_repeated = evaluate_m0_repeated(
            model,
            validation_samples,
            masker,
            validation_mask_seeds,
            batch_size=batch_size,
            device=device,
            loss_weights=loss_weights,
            groups=groups,
        )
        validation = validation_repeated["aggregate"]
        validation["macro_family_mse"] = validation["macro_family_mse_mean"]
        history.append(
            {
                "epoch": epoch,
                "training_batch_loss": epoch_loss / max(update_count, 1),
                "validation": validation,
                "validation_mask_seeds": list(validation_mask_seeds),
                "selection_metric": selection_metric,
                "selection_metric_value": float(validation[aggregate_selection_key]),
            }
        )
        metric = float(validation[aggregate_selection_key])
        if metric < best_metric:
            best_metric = metric
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
