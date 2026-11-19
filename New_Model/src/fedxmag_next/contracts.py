"""Tensor contracts and safe loading for graph windows.

This module is independent from the archived package.  It can read its
tensor-only artifact format while validating every shape at the boundary.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class GraphWindow:
    """A single, score-time graph snapshot.

    Values for unavailable features must still be finite.  Availability flags,
    rather than NaNs, carry missingness semantics.
    """

    node_features: Tensor
    edge_features: Tensor
    edge_index: Tensor
    node_available: Tensor
    edge_available: Tensor
    label: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for kind in ("node", "edge"):
            values = getattr(self, f"{kind}_features")
            available = getattr(self, f"{kind}_available")
            if values.ndim != 2 or values.numel() == 0:
                raise ValueError(f"{kind}_features must be a nonempty rank-2 tensor")
            if available.shape != values.shape or available.dtype is not torch.bool:
                raise ValueError(f"{kind}_available must be a bool tensor matching features")
            if not torch.isfinite(values).all():
                raise ValueError(f"{kind}_features contains non-finite values")
        if self.edge_index.dtype is not torch.long or self.edge_index.shape != (
            2,
            self.edge_features.shape[0],
        ):
            raise ValueError("edge_index must be [2, edges] torch.long")
        if self.edge_index.numel() and (
            int(self.edge_index.min()) < 0
            or int(self.edge_index.max()) >= self.node_features.shape[0]
        ):
            raise ValueError("edge_index contains an out-of-range node")
        if self.label not in {None, 0, 1}:
            raise ValueError("label must be None, 0, or 1")

    @property
    def node_dim(self) -> int:
        return int(self.node_features.shape[1])

    @property
    def edge_dim(self) -> int:
        return int(self.edge_features.shape[1])

    def timestamp(self) -> int | None:
        value = self.metadata.get("window_start_ns")
        return int(value) if value is not None else None

    def keys(self, kind: str) -> tuple[str, ...] | None:
        """Stable entity keys for causal state; ``None`` means no temporal link."""

        field_name = "endpoint_keys" if kind == "node" else "edge_keys"
        values = self.metadata.get(field_name)
        expected = self.node_features.shape[0] if kind == "node" else self.edge_features.shape[0]
        if values is None or len(values) != expected:
            return None
        if kind == "edge":
            return tuple("\u2192".join(map(str, value)) for value in values)
        return tuple(map(str, values))


@dataclass(frozen=True)
class GraphBatch:
    node_features: Tensor
    edge_features: Tensor
    edge_index: Tensor
    node_available: Tensor
    edge_available: Tensor
    node_graph: Tensor
    edge_graph: Tensor
    size: int

    def to(self, device: torch.device | str) -> GraphBatch:
        return GraphBatch(
            self.node_features.to(device),
            self.edge_features.to(device),
            self.edge_index.to(device),
            self.node_available.to(device),
            self.edge_available.to(device),
            self.node_graph.to(device),
            self.edge_graph.to(device),
            self.size,
        )


@dataclass(frozen=True)
class MaskedBatch:
    """A batch with corruption targets retained outside encoder-visible input."""

    batch: GraphBatch
    node_target: Tensor
    edge_target: Tensor
    node_mask: Tensor
    edge_mask: Tensor


def collate(windows: Iterable[GraphWindow]) -> GraphBatch:
    items = list(windows)
    if not items:
        raise ValueError("Cannot collate an empty sequence")
    node_dim, edge_dim = items[0].node_dim, items[0].edge_dim
    all_nodes: list[Tensor] = []
    all_edges: list[Tensor] = []
    all_indices: list[Tensor] = []
    node_available: list[Tensor] = []
    edge_available: list[Tensor] = []
    node_graph: list[Tensor] = []
    edge_graph: list[Tensor] = []
    offset = 0
    for graph_id, item in enumerate(items):
        item.validate()
        if item.node_dim != node_dim or item.edge_dim != edge_dim:
            raise ValueError("Every graph in a batch must share feature dimensions")
        nodes, edges = item.node_features.shape[0], item.edge_features.shape[0]
        all_nodes.append(item.node_features)
        all_edges.append(item.edge_features)
        all_indices.append(item.edge_index + offset)
        node_available.append(item.node_available)
        edge_available.append(item.edge_available)
        node_graph.append(torch.full((nodes,), graph_id, dtype=torch.long))
        edge_graph.append(torch.full((edges,), graph_id, dtype=torch.long))
        offset += nodes
    return GraphBatch(
        torch.cat(all_nodes),
        torch.cat(all_edges),
        torch.cat(all_indices, dim=1),
        torch.cat(node_available),
        torch.cat(edge_available),
        torch.cat(node_graph),
        torch.cat(edge_graph),
        len(items),
    )


def mask_features(batch: GraphBatch, rate: float, generator: torch.Generator) -> MaskedBatch:
    """Artificially mask only naturally available values for pretext training."""

    if not 0.0 < rate < 1.0:
        raise ValueError("mask rate must lie strictly between zero and one")

    def corrupt(values: Tensor, available: Tensor) -> tuple[Tensor, Tensor]:
        draw = torch.rand(values.shape, generator=generator, device=values.device)
        mask = available & (draw < rate)
        # Guarantee a target whenever a graph/entity group happens to have values.
        if available.any() and not mask.any():
            mask[torch.nonzero(available, as_tuple=False)[0].unbind()] = True
        return values.masked_fill(mask, 0.0), mask

    node_values, node_mask = corrupt(batch.node_features, batch.node_available)
    edge_values, edge_mask = corrupt(batch.edge_features, batch.edge_available)
    visible = GraphBatch(
        node_values,
        edge_values,
        batch.edge_index,
        batch.node_available,
        batch.edge_available,
        batch.node_graph,
        batch.edge_graph,
        batch.size,
    )
    return MaskedBatch(visible, batch.node_features, batch.edge_features, node_mask, edge_mask)


def clean_batch(batch: GraphBatch) -> MaskedBatch:
    return MaskedBatch(
        batch,
        batch.node_features,
        batch.edge_features,
        torch.zeros_like(batch.node_available),
        torch.zeros_like(batch.edge_available),
    )


def load_windows(path: str | Path) -> list[GraphWindow]:
    """Load the archived tensor-only graph artifact without importing archived code."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("Expected a tensor-only graph artifact with a records list")
    needed = {
        "node_features",
        "edge_features",
        "edge_index",
        "node_available",
        "edge_available",
        "label",
        "metadata_json",
    }
    result: list[GraphWindow] = []
    for index, record in enumerate(payload["records"]):
        if not isinstance(record, dict) or set(record) != needed:
            raise ValueError(f"Invalid graph record at position {index}")
        try:
            metadata = json.loads(record["metadata_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid metadata in graph record {index}") from error
        item = GraphWindow(
            record["node_features"].float(),
            record["edge_features"].float(),
            record["edge_index"].long(),
            record["node_available"].bool(),
            record["edge_available"].bool(),
            int(record["label"]),
            metadata,
        )
        item.validate()
        result.append(item)
    return result


def require_normal(windows: Iterable[GraphWindow], role: str) -> list[GraphWindow]:
    items = list(windows)
    if not items or any(item.label != 0 for item in items):
        raise ValueError(f"{role} must contain at least one externally vetted normal window")
    return items
