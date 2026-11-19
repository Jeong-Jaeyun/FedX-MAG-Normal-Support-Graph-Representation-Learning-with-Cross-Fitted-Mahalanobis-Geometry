"""Separately normalized M0 target-family losses."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from fedxmag.masking import MaskedGraphBatch
from fedxmag.models.m0 import M0Output
from fedxmag.schema import DEFAULT_FEATURE_GROUPS, FeatureGroups

FAMILY_KEYS = ("node_timing", "node_distribution", "edge_timing", "edge_distribution")


@dataclass
class M0Loss:
    total: Tensor
    components: dict[str, Tensor]
    counts: dict[str, int]
    graph_counts: dict[str, int]


def _graph_balanced_masked_mse(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    graph_index: Tensor,
    num_graphs: int,
) -> tuple[Tensor, int, int]:
    count = int(mask.sum())
    if count == 0:
        return prediction.sum() * 0.0, 0, 0
    squared_error = (prediction - target).square()
    graph_losses: list[Tensor] = []
    for graph in range(num_graphs):
        graph_mask = mask[graph_index == graph]
        if bool(graph_mask.any()):
            selected_error = squared_error[graph_index == graph].masked_select(graph_mask)
            graph_losses.append(selected_error.mean())
    return torch.stack(graph_losses).mean(), count, len(graph_losses)


def m0_loss(
    output: M0Output,
    masked: MaskedGraphBatch,
    weights: Mapping[str, float] | None = None,
    groups: FeatureGroups = DEFAULT_FEATURE_GROUPS,
) -> M0Loss:
    active_weights = {key: 1.0 for key in FAMILY_KEYS}
    if weights is not None:
        unknown = set(weights) - set(FAMILY_KEYS)
        if unknown:
            raise ValueError(f"Unknown loss weights: {sorted(unknown)}")
        active_weights.update(weights)

    components: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    graph_counts: dict[str, int] = {}
    for kind, prediction, target, targeted, graph_index in (
        (
            "node",
            output.node_prediction,
            masked.node_targets,
            masked.node_targeted,
            masked.batch.node_graph_index,
        ),
        (
            "edge",
            output.edge_prediction,
            masked.edge_targets,
            masked.edge_targeted,
            masked.batch.edge_graph_index,
        ),
    ):
        for family in ("timing", "distribution"):
            key = f"{kind}_{family}"
            indices = list(groups.indices(kind, family))
            value, count, graph_count = _graph_balanced_masked_mse(
                prediction[:, indices],
                target[:, indices],
                targeted[:, indices],
                graph_index,
                masked.batch.num_graphs,
            )
            components[key] = value
            counts[key] = count
            graph_counts[key] = graph_count

    total = torch.stack(
        [components[key] * active_weights[key] for key in FAMILY_KEYS]
    ).sum()
    return M0Loss(
        total=total,
        components=components,
        counts=counts,
        graph_counts=graph_counts,
    )
