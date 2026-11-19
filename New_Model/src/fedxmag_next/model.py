"""Causal, edge-aware graph encoder with entity-level outputs.

There is intentionally no graph pooling head in this architecture.  A window
can be summarized for logging, but anomaly evidence stays attached to nodes and
directed edges until tail aggregation in the detector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .contracts import MaskedBatch


def segment_mean(values: Tensor, index: Tensor, groups: int) -> Tensor:
    result = values.new_zeros((groups, values.shape[1]))
    counts = values.new_zeros((groups, 1))
    result.index_add_(0, index, values)
    counts.index_add_(0, index, torch.ones_like(values[:, :1]))
    return result / counts.clamp_min(1.0)


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.layers(values)


class DirectedEdgeLayer(nn.Module):
    """Incoming/outgoing messages plus edge updates; no within-window graph mean."""

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.edge_message = FeedForward(3 * hidden, hidden, hidden, dropout)
        self.in_update = FeedForward(2 * hidden, hidden, hidden, dropout)
        self.out_update = FeedForward(2 * hidden, hidden, hidden, dropout)
        self.node_update = FeedForward(3 * hidden, hidden, hidden, dropout)
        self.edge_update = FeedForward(3 * hidden, hidden, hidden, dropout)
        self.node_norm = nn.LayerNorm(hidden)
        self.edge_norm = nn.LayerNorm(hidden)

    def forward(self, node: Tensor, edge: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        source, destination = edge_index
        message = self.edge_message(torch.cat((node[source], edge, node[destination]), dim=1))
        incoming = segment_mean(message, destination, node.shape[0])
        outgoing = segment_mean(message, source, node.shape[0])
        node_delta = self.node_update(
            torch.cat(
                (
                    node,
                    self.in_update(torch.cat((node, incoming), dim=1)),
                    self.out_update(torch.cat((node, outgoing), dim=1)),
                ),
                dim=1,
            )
        )
        next_node = self.node_norm(node + node_delta)
        edge_delta = self.edge_update(
            torch.cat((edge, next_node[source], next_node[destination]), dim=1)
        )
        return next_node, self.edge_norm(edge + edge_delta)


@dataclass(frozen=True)
class EncoderConfig:
    node_dim: int
    edge_dim: int
    hidden_dim: int = 64
    layers: int = 3
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if min(self.node_dim, self.edge_dim, self.hidden_dim, self.layers) < 1:
            raise ValueError("encoder dimensions and layers must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class EncoderOutput:
    node_embedding: Tensor
    edge_embedding: Tensor
    node_prediction: Tensor
    edge_prediction: Tensor


class CausalEdgeGraphEncoder(nn.Module):
    """Encoder that accepts only prior keyed state as temporal context.

    ``node_memory`` and ``edge_memory`` are optional tensors aligned to the
    current entities.  They are supplied by :class:`CausalStateStore`; the
    model never retains or mutates score-time state internally.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.node_input = nn.Linear(2 * config.node_dim, config.hidden_dim)
        self.edge_input = nn.Linear(2 * config.edge_dim, config.hidden_dim)
        self.node_time = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.edge_time = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            DirectedEdgeLayer(config.hidden_dim, config.dropout) for _ in range(config.layers)
        )
        self.node_decoder = FeedForward(
            config.hidden_dim, config.hidden_dim, config.node_dim, config.dropout
        )
        self.edge_decoder = FeedForward(
            config.hidden_dim, config.hidden_dim, config.edge_dim, config.dropout
        )

    def forward(
        self,
        masked: MaskedBatch,
        *,
        node_memory: Tensor | None = None,
        edge_memory: Tensor | None = None,
    ) -> EncoderOutput:
        batch = masked.batch
        node = self.node_input(
            torch.cat(
                (batch.node_features * batch.node_available, batch.node_available.float()), dim=1
            )
        )
        edge = self.edge_input(
            torch.cat(
                (batch.edge_features * batch.edge_available, batch.edge_available.float()), dim=1
            )
        )
        if node_memory is not None:
            if node_memory.shape != node.shape:
                raise ValueError("node memory must align with node encoder states")
            node = self.node_time(node, node_memory)
        if edge_memory is not None:
            if edge_memory.shape != edge.shape:
                raise ValueError("edge memory must align with edge encoder states")
            edge = self.edge_time(edge, edge_memory)
        for layer in self.layers:
            node, edge = layer(node, edge, batch.edge_index)
        return EncoderOutput(node, edge, self.node_decoder(node), self.edge_decoder(edge))
