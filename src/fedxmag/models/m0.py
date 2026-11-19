"""Static edge-aware encoder and M0 reconstruction baselines."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from fedxmag.masking import MaskedGraphBatch


def segment_mean(values: Tensor, index: Tensor, num_segments: int) -> Tensor:
    if values.ndim != 2:
        raise ValueError("segment_mean expects a rank-2 value tensor")
    result = values.new_zeros((num_segments, values.shape[1]))
    counts = values.new_zeros((num_segments, 1))
    if values.shape[0]:
        result.index_add_(0, index, values)
        counts.index_add_(0, index, values.new_ones((values.shape[0], 1)))
    return result / counts.clamp_min(1.0)


class MLP(nn.Module):
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


class EdgeAwareLayer(nn.Module):
    """Separate incoming/outgoing directed aggregation with edge-state updates."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.in_message = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.out_message = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.node_update = MLP(3 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.edge_update = MLP(3 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, node_state: Tensor, edge_state: Tensor, edge_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        source, destination = edge_index
        incoming_message = self.in_message(torch.cat((node_state[source], edge_state), dim=1))
        outgoing_message = self.out_message(
            torch.cat((node_state[destination], edge_state), dim=1)
        )
        incoming = segment_mean(incoming_message, destination, node_state.shape[0])
        outgoing = segment_mean(outgoing_message, source, node_state.shape[0])
        node_delta = self.node_update(torch.cat((node_state, incoming, outgoing), dim=1))
        updated_node = self.node_norm(node_state + node_delta)
        edge_delta = self.edge_update(
            torch.cat((edge_state, updated_node[source], updated_node[destination]), dim=1)
        )
        updated_edge = self.edge_norm(edge_state + edge_delta)
        return updated_node, updated_edge


class SetContextLayer(nn.Module):
    """Capacity-matched set layer that never reads endpoint adjacency."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        # These four MLPs exactly match EdgeAwareLayer's parameter shapes.
        self.node_context_a = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.node_context_b = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.node_update = MLP(3 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.edge_update = MLP(3 * hidden_dim, hidden_dim, hidden_dim, dropout)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_state: Tensor,
        edge_state: Tensor,
        node_graph_index: Tensor,
        edge_graph_index: Tensor,
        num_graphs: int,
    ) -> tuple[Tensor, Tensor]:
        pooled_node = segment_mean(node_state, node_graph_index, num_graphs)
        pooled_edge = segment_mean(edge_state, edge_graph_index, num_graphs)
        set_state = torch.cat((pooled_node, pooled_edge), dim=1)
        context_a = self.node_context_a(set_state)
        context_b = self.node_context_b(set_state)
        node_delta = self.node_update(
            torch.cat(
                (
                    node_state,
                    context_a[node_graph_index],
                    context_b[node_graph_index],
                ),
                dim=1,
            )
        )
        updated_node = self.node_norm(node_state + node_delta)
        updated_node_pool = segment_mean(updated_node, node_graph_index, num_graphs)
        edge_delta = self.edge_update(
            torch.cat(
                (
                    edge_state,
                    updated_node_pool[edge_graph_index],
                    pooled_edge[edge_graph_index],
                ),
                dim=1,
            )
        )
        updated_edge = self.edge_norm(edge_state + edge_delta)
        return updated_node, updated_edge


@dataclass
class M0Output:
    node_prediction: Tensor
    edge_prediction: Tensor
    graph_embedding: Tensor


def _encoder_values(masked: MaskedGraphBatch, kind: str) -> Tensor:
    visible = getattr(masked, f"{kind}_visible")
    available = getattr(masked.batch, f"{kind}_available")
    corrupted = getattr(masked, f"{kind}_corrupted")
    return torch.cat((visible, available.to(visible.dtype), corrupted.to(visible.dtype)), dim=1)


class M0GraphModel(nn.Module):
    """Minimal static graph model with node/edge decoders and graph embedding."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        graph_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        self.node_input = nn.Linear(3 * node_dim, hidden_dim)
        self.edge_input = nn.Linear(3 * edge_dim, hidden_dim)
        self.layers = nn.ModuleList(
            EdgeAwareLayer(hidden_dim, dropout) for _ in range(num_layers)
        )
        self.graph_projection = MLP(2 * hidden_dim, hidden_dim, graph_dim, dropout)
        self.node_decoder = MLP(hidden_dim + graph_dim, hidden_dim, node_dim, dropout)
        self.edge_decoder = MLP(3 * hidden_dim + graph_dim, hidden_dim, edge_dim, dropout)

    def forward(self, masked: MaskedGraphBatch) -> M0Output:
        batch = masked.batch
        node_state = self.node_input(_encoder_values(masked, "node"))
        edge_state = self.edge_input(_encoder_values(masked, "edge"))
        for layer in self.layers:
            node_state, edge_state = layer(node_state, edge_state, batch.edge_index)

        pooled_node = segment_mean(node_state, batch.node_graph_index, batch.num_graphs)
        pooled_edge = segment_mean(edge_state, batch.edge_graph_index, batch.num_graphs)
        graph_embedding = self.graph_projection(torch.cat((pooled_node, pooled_edge), dim=1))

        node_graph = graph_embedding[batch.node_graph_index]
        source, destination = batch.edge_index
        edge_graph = graph_embedding[batch.edge_graph_index]
        node_prediction = self.node_decoder(torch.cat((node_state, node_graph), dim=1))
        edge_prediction = self.edge_decoder(
            torch.cat(
                (node_state[source], node_state[destination], edge_state, edge_graph), dim=1
            )
        )
        return M0Output(node_prediction, edge_prediction, graph_embedding)


class SetContextPredictor(nn.Module):
    """Graph-capacity control using unordered within-window set context only."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        graph_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        self.node_input = nn.Linear(3 * node_dim, hidden_dim)
        self.edge_input = nn.Linear(3 * edge_dim, hidden_dim)
        self.layers = nn.ModuleList(
            SetContextLayer(hidden_dim, dropout) for _ in range(num_layers)
        )
        self.graph_projection = MLP(2 * hidden_dim, hidden_dim, graph_dim, dropout)
        self.node_decoder = MLP(hidden_dim + graph_dim, hidden_dim, node_dim, dropout)
        self.edge_decoder = MLP(3 * hidden_dim + graph_dim, hidden_dim, edge_dim, dropout)

    def forward(self, masked: MaskedGraphBatch) -> M0Output:
        batch = masked.batch
        node_state = self.node_input(_encoder_values(masked, "node"))
        edge_state = self.edge_input(_encoder_values(masked, "edge"))
        for layer in self.layers:
            node_state, edge_state = layer(
                node_state,
                edge_state,
                batch.node_graph_index,
                batch.edge_graph_index,
                batch.num_graphs,
            )

        pooled_node = segment_mean(node_state, batch.node_graph_index, batch.num_graphs)
        pooled_edge = segment_mean(edge_state, batch.edge_graph_index, batch.num_graphs)
        graph_embedding = self.graph_projection(torch.cat((pooled_node, pooled_edge), dim=1))
        node_graph = graph_embedding[batch.node_graph_index]
        edge_graph = graph_embedding[batch.edge_graph_index]
        node_prediction = self.node_decoder(torch.cat((node_state, node_graph), dim=1))
        edge_prediction = self.edge_decoder(
            torch.cat(
                (
                    pooled_node[batch.edge_graph_index],
                    pooled_edge[batch.edge_graph_index],
                    edge_state,
                    edge_graph,
                ),
                dim=1,
            )
        )
        return M0Output(node_prediction, edge_prediction, graph_embedding)


def _nonidentity_permutation(count: int, generator: torch.Generator) -> Tensor:
    permutation = torch.randperm(count, generator=generator)
    if torch.equal(permutation, torch.arange(count)):
        permutation = torch.roll(permutation, shifts=1)
    return permutation


def _shuffle_endpoints_within_graph(
    masked: MaskedGraphBatch, seed: int, mode: str
) -> MaskedGraphBatch:
    """Preserve graph membership and degree marginals while breaking endpoint pairing."""

    if mode not in {"destination_only", "both_endpoints"}:
        raise ValueError(f"Unknown endpoint shuffle mode: {mode}")

    batch = masked.batch
    source, destination = batch.edge_index
    shuffled_source = source.clone()
    shuffled_destination = destination.clone()
    for graph in range(batch.num_graphs):
        rows = torch.nonzero(batch.edge_graph_index == graph, as_tuple=False).flatten()
        count = int(rows.numel())
        if count < 2:
            continue
        destination_generator = torch.Generator().manual_seed(
            seed + graph * 1_000_003 + count
        )
        destination_permutation = _nonidentity_permutation(
            count, destination_generator
        ).to(rows.device)
        shuffled_destination[rows] = destination[rows[destination_permutation]]
        if mode == "both_endpoints":
            source_generator = torch.Generator().manual_seed(
                seed + 104_729 + graph * 1_000_003 + count
            )
            source_permutation = _nonidentity_permutation(count, source_generator).to(
                rows.device
            )
            shuffled_source[rows] = source[rows[source_permutation]]
    shuffled_batch = replace(
        batch,
        edge_index=torch.stack((shuffled_source, shuffled_destination)),
    )
    return replace(masked, batch=shuffled_batch)


class AdjacencyShuffledGraphModel(M0GraphModel):
    """Exact-capacity graph control trained and evaluated on shuffled endpoints."""

    def __init__(
        self,
        *args: object,
        shuffle_seed: int = 271_828,
        endpoint_shuffle_mode: str = "destination_only",
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.shuffle_seed = int(shuffle_seed)
        if endpoint_shuffle_mode not in {"destination_only", "both_endpoints"}:
            raise ValueError(f"Unknown endpoint shuffle mode: {endpoint_shuffle_mode}")
        self.endpoint_shuffle_mode = endpoint_shuffle_mode

    def forward(self, masked: MaskedGraphBatch) -> M0Output:
        return super().forward(
            _shuffle_endpoints_within_graph(
                masked, self.shuffle_seed, self.endpoint_shuffle_mode
            )
        )


class LocalPredictor(nn.Module):
    """Non-graph admissibility baseline using only each entity's visible row."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_decoder = MLP(3 * node_dim, hidden_dim, node_dim, dropout)
        self.edge_decoder = MLP(3 * edge_dim, hidden_dim, edge_dim, dropout)

    def forward(self, masked: MaskedGraphBatch) -> M0Output:
        node_prediction = self.node_decoder(_encoder_values(masked, "node"))
        edge_prediction = self.edge_decoder(_encoder_values(masked, "edge"))
        graph_embedding = node_prediction.new_zeros((masked.batch.num_graphs, 0))
        return M0Output(node_prediction, edge_prediction, graph_embedding)


class ConstantPredictor(nn.Module):
    """Training-normal mean predictor; standardized targets make the mean zero."""

    def forward(self, masked: MaskedGraphBatch) -> M0Output:
        return M0Output(
            node_prediction=torch.zeros_like(masked.node_targets),
            edge_prediction=torch.zeros_like(masked.edge_targets),
            graph_embedding=masked.node_targets.new_zeros((masked.batch.num_graphs, 0)),
        )
