"""Joint M1 training (M0 reconstruction + cross-fitted normal support)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from fedxmag.confirmation import build_masked_batches, per_graph_family_errors
from fedxmag.losses import FAMILY_KEYS, m0_loss
from fedxmag.m1.covariance import spectral_diagnostics
from fedxmag.m1.objective import NormalSupportSettings, normal_support_loss
from fedxmag.masking import FamilyMasker, MaskedGraphBatch
from fedxmag.schema import FeatureGroups, GraphBatch, GraphSample, collate_graphs

SELECTION_MODES = ("reconstruction", "normal_support")


def clean_view(batch: GraphBatch) -> MaskedGraphBatch:
    """Score-time encoder view: natural availability only, no artificial corruption."""

    node_none = torch.zeros_like(batch.node_available)
    edge_none = torch.zeros_like(batch.edge_available)
    return MaskedGraphBatch(
        batch=batch,
        node_visible=batch.node_features.masked_fill(~batch.node_available, 0.0),
        edge_visible=batch.edge_features.masked_fill(~batch.edge_available, 0.0),
        node_corrupted=node_none,
        edge_corrupted=edge_none,
        node_targeted=node_none.clone(),
        edge_targeted=edge_none.clone(),
    )


@torch.no_grad()
def embed_clean(
    model: nn.Module,
    samples: Sequence[GraphSample],
    groups: FeatureGroups,
    batch_size: int = 64,
) -> Tensor:
    model.eval()
    parts = [
        model(clean_view(collate_graphs(list(samples[start : start + batch_size]), groups)))
        .graph_embedding.to(torch.float64)
        for start in range(0, len(samples), batch_size)
    ]
    return torch.cat(parts, dim=0)


def _family_means(errors: Sequence[np.ndarray]) -> np.ndarray:
    """Mask-mean of graph-balanced family MSE, as in ``evaluate_m0_repeated``."""

    return np.mean([np.nanmean(item, axis=0) for item in errors], axis=0)


@torch.no_grad()
def _validation(
    model: nn.Module,
    cached: Sequence[Sequence[MaskedGraphBatch]],
    settings: NormalSupportSettings,
    split_seeds: Sequence[int],
    groups: FeatureGroups,
    clean_samples: Sequence[GraphSample],
) -> dict[str, Any]:
    model.eval()
    errors = [per_graph_family_errors(model, batches, groups) for batches in cached]
    family = _family_means(errors)
    support_values: list[float] = []
    for batches, split_seed in zip(cached, split_seeds, strict=True):
        embeddings = torch.cat([model(masked).graph_embedding for masked in batches], dim=0)
        if embeddings.shape[1] == 0:
            continue
        loss = normal_support_loss(
            embeddings, settings, torch.Generator().manual_seed(split_seed)
        )
        support_values.append(float(loss.total))
    clean = embed_clean(model, clean_samples, groups)
    geometry = spectral_diagnostics(torch.cov(clean.T)) if clean.shape[1] > 1 else {}
    return {
        **{f"{key}_mse": float(value) for key, value in zip(FAMILY_KEYS, family, strict=True)},
        "worst_family_mse": float(np.max(family)),
        "macro_family_mse": float(np.mean(family)),
        "normal_support_objective": float(np.mean(support_values))
        if support_values
        else float("nan"),
        "clean_select_geometry": geometry,
    }


def train_m1(
    model: nn.Module,
    train_samples: Sequence[GraphSample],
    select_samples: Sequence[GraphSample],
    masker: FamilyMasker,
    settings: NormalSupportSettings,
    *,
    epochs: int,
    batch_size: int,
    validation_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    early_stopping_patience: int,
    validation_mask_seeds: Sequence[int],
    selection_mode: str,
    loss_weights: Mapping[str, float],
    seed: int,
    groups: FeatureGroups,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Train with drop-last batches so every cross-fit half has ``batch_size // 2`` rows."""

    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"Unknown selection mode: {selection_mode}")
    if settings.reconstruction_weight == 0 and selection_mode == "reconstruction":
        raise ValueError("Reconstruction-free training cannot select on reconstruction")
    if len(train_samples) < batch_size:
        raise ValueError("Training needs at least one full batch")
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    split_generator = torch.Generator().manual_seed(seed + 7_919)
    cached = [
        build_masked_batches(select_samples, masker, mask_seed, validation_batch_size, groups)
        for mask_seed in validation_mask_seeds
    ]
    split_seeds = [mask_seed + 104_729 for mask_seed in validation_mask_seeds]
    key = "worst_family_mse" if selection_mode == "reconstruction" else "normal_support_objective"
    best = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    full_batches = len(train_samples) // batch_size

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_samples), generator=generator).tolist()
        sums: dict[str, float] = {}
        for index in range(full_batches):
            chosen = order[index * batch_size : (index + 1) * batch_size]
            batch = collate_graphs([train_samples[item] for item in chosen], groups).to(device)
            masked = masker(batch, generator=generator)
            output = model(masked)
            reconstruction = m0_loss(output, masked, weights=loss_weights, groups=groups)
            total = settings.reconstruction_weight * reconstruction.total
            record = {"reconstruction": float(reconstruction.total.detach())}
            if settings.uses_normal_support:
                support = normal_support_loss(output.graph_embedding, settings, split_generator)
                total = total + support.total.to(total.dtype)
                record.update(support.diagnostics)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            record["total"] = float(total.detach())
            for name, value in record.items():
                sums[name] = sums.get(name, 0.0) + value
        training = {name: value / full_batches for name, value in sums.items()}
        validation = _validation(model, cached, settings, split_seeds, groups, select_samples)
        metric = float(validation[key])
        history.append(
            {
                "epoch": epoch,
                "training": training,
                "validation": validation,
                "selection_mode": selection_mode,
                "selection_metric_value": metric,
            }
        )
        if np.isfinite(metric) and metric < best:
            best = metric
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= early_stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
