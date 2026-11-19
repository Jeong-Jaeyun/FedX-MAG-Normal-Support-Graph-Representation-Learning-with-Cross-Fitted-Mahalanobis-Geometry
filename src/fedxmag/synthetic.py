"""Deterministic synthetic graphs for software verification only."""

from __future__ import annotations

import torch

from fedxmag.schema import DEFAULT_FEATURE_GROUPS, GraphSample


def make_synthetic_graphs(
    num_graphs: int = 32,
    min_nodes: int = 5,
    max_nodes: int = 9,
    seed: int = 7,
) -> list[GraphSample]:
    if min_nodes < 2 or max_nodes < min_nodes:
        raise ValueError("Expected 2 <= min_nodes <= max_nodes")
    generator = torch.Generator().manual_seed(seed)
    samples: list[GraphSample] = []
    groups = DEFAULT_FEATURE_GROUPS

    for graph_id in range(num_graphs):
        num_nodes = int(torch.randint(min_nodes, max_nodes + 1, (), generator=generator))
        source = torch.arange(num_nodes, dtype=torch.long).repeat(2)
        destination = torch.cat(
            (
                (torch.arange(num_nodes) + 1) % num_nodes,
                (torch.arange(num_nodes) - 1) % num_nodes,
            )
        ).long()
        edge_index = torch.stack((source, destination))
        graph_context = torch.randn((1, 3), generator=generator)
        node_latent = torch.randn((num_nodes, 3), generator=generator) + graph_context
        edge_latent = (
            0.6 * node_latent[source]
            + 0.4 * node_latent[destination]
            + 0.05 * torch.randn((source.shape[0], 3), generator=generator)
        )

        node_basis = torch.cat(
            (node_latent, node_latent.square(), torch.sin(node_latent)), dim=1
        )
        edge_basis = torch.cat(
            (edge_latent, edge_latent.square(), torch.cos(edge_latent)), dim=1
        )
        node_features = torch.cat(
            (
                node_basis[:, :8],
                0.7 * node_basis[:, :6] + 0.3 * node_basis[:, 3:9],
            ),
            dim=1,
        )
        edge_features = torch.cat(
            (edge_basis[:, :4], 0.5 * edge_basis[:, :3] + 0.5 * edge_basis[:, 3:6]),
            dim=1,
        )
        node_features += 0.02 * torch.randn(node_features.shape, generator=generator)
        edge_features += 0.02 * torch.randn(edge_features.shape, generator=generator)

        node_available = torch.rand(node_features.shape, generator=generator) > 0.03
        edge_available = torch.rand(edge_features.shape, generator=generator) > 0.03
        node_features = node_features.masked_fill(~node_available, 0.0)
        edge_features = edge_features.masked_fill(~edge_available, 0.0)
        sample = GraphSample(
            node_features=node_features[:, : groups.node_dim],
            edge_features=edge_features[:, : groups.edge_dim],
            edge_index=edge_index,
            node_available=node_available[:, : groups.node_dim],
            edge_available=edge_available[:, : groups.edge_dim],
            label=0,
            metadata={"synthetic_graph_id": graph_id},
        )
        sample.validate(groups)
        samples.append(sample)
    return samples
