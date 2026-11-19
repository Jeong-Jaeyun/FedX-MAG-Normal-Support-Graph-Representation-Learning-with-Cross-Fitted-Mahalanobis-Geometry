import torch

from fedxmag.schema import DEFAULT_FEATURE_GROUPS, GraphSample, collate_graphs


def _sample(node_count: int) -> GraphSample:
    groups = DEFAULT_FEATURE_GROUPS
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return GraphSample(
        node_features=torch.zeros((node_count, groups.node_dim)),
        edge_features=torch.zeros((2, groups.edge_dim)),
        edge_index=edge_index,
        node_available=torch.ones((node_count, groups.node_dim), dtype=torch.bool),
        edge_available=torch.ones((2, groups.edge_dim), dtype=torch.bool),
    )


def test_collate_offsets_edges_and_tracks_graphs() -> None:
    batch = collate_graphs([_sample(2), _sample(3)])
    assert batch.num_graphs == 2
    assert batch.edge_index.tolist() == [[0, 1, 2, 3], [1, 0, 3, 2]]
    assert batch.node_graph_index.tolist() == [0, 0, 1, 1, 1]
    assert batch.edge_graph_index.tolist() == [0, 0, 1, 1]

