import torch

from fedxmag.masking import FamilyMasker
from fedxmag.schema import DEFAULT_FEATURE_GROUPS, GraphSample, collate_graphs
from fedxmag.synthetic import make_synthetic_graphs


def test_family_masking_respects_availability_and_graph_local_budget() -> None:
    groups = DEFAULT_FEATURE_GROUPS
    batch = collate_graphs(make_synthetic_graphs(num_graphs=4, seed=11))
    masker = FamilyMasker(
        attribute_budget=0.30,
        family_mask_probability=1.0,
        minimum_visible_per_entity=1,
    )
    masked = masker(batch, generator=torch.Generator().manual_seed(3))

    assert torch.all(masked.node_corrupted <= batch.node_available)
    assert torch.all(masked.edge_corrupted <= batch.edge_available)
    assert int(masked.node_corrupted.sum() + masked.edge_corrupted.sum()) > 0
    assert torch.all(masked.node_targeted <= masked.node_corrupted)
    assert torch.all(masked.edge_targeted <= masked.edge_corrupted)

    for graph in range(batch.num_graphs):
        node_rows = batch.node_graph_index == graph
        edge_rows = batch.edge_graph_index == graph
        masked_count = int(
            masked.node_corrupted[node_rows].sum()
            + masked.edge_corrupted[edge_rows].sum()
        )
        available_count = int(
            batch.node_available[node_rows].sum() + batch.edge_available[edge_rows].sum()
        )
        assert masked_count <= int(0.30 * available_count)

    source, destination = batch.edge_index
    for edge in range(batch.edge_features.shape[0]):
        for family in ("timing", "distribution"):
            edge_indices = list(groups.indices("edge", family))
            if bool(masked.edge_targeted[edge, edge_indices].any()):
                out_indices = list(groups.directional_node_indices(family, "out"))
                in_indices = list(groups.directional_node_indices(family, "in"))
                assert torch.equal(
                    masked.node_corrupted[source[edge], out_indices],
                    batch.node_available[source[edge], out_indices],
                )
                assert torch.equal(
                    masked.node_corrupted[destination[edge], in_indices],
                    batch.node_available[destination[edge], in_indices],
                )

    assert bool(masked.node_targeted.any())
    for node in range(batch.node_features.shape[0]):
        for family in ("timing", "distribution"):
            edge_indices = list(groups.indices("edge", family))
            for direction in ("out", "in"):
                node_indices = list(groups.directional_node_indices(family, direction))
                if not bool(masked.node_targeted[node, node_indices].any()):
                    continue
                related = (source == node) if direction == "out" else (destination == node)
                assert torch.equal(
                    masked.edge_corrupted[related][:, edge_indices],
                    batch.edge_available[related][:, edge_indices],
                )
                related_edges = torch.nonzero(related, as_tuple=False).flatten()
                out_indices = list(groups.directional_node_indices(family, "out"))
                in_indices = list(groups.directional_node_indices(family, "in"))
                for edge in related_edges:
                    assert torch.equal(
                        masked.node_corrupted[source[edge], out_indices],
                        batch.node_available[source[edge], out_indices],
                    )
                    assert torch.equal(
                        masked.node_corrupted[destination[edge], in_indices],
                        batch.node_available[destination[edge], in_indices],
                    )


def test_zero_mask_probability_really_produces_no_targets() -> None:
    batch = collate_graphs(make_synthetic_graphs(num_graphs=2, seed=5))
    masked = FamilyMasker(0.30, 0.0)(batch, torch.Generator().manual_seed(3))
    assert not bool(masked.node_corrupted.any())
    assert not bool(masked.edge_corrupted.any())
    assert not bool(masked.node_targeted.any())
    assert not bool(masked.edge_targeted.any())


def test_sparse_entity_can_be_fully_attribute_masked_with_graph_context() -> None:
    groups = DEFAULT_FEATURE_GROUPS
    node_features = torch.zeros((2, groups.node_dim))
    node_available = torch.zeros_like(node_features, dtype=torch.bool)
    node_available[0, :4] = True
    node_available[1, 8] = True
    sample = GraphSample(
        node_features=node_features,
        edge_features=torch.empty((0, groups.edge_dim)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        node_available=node_available,
        edge_available=torch.empty((0, groups.edge_dim), dtype=torch.bool),
    )
    batch = collate_graphs([sample])
    masked = FamilyMasker(
        attribute_budget=0.80,
        family_mask_probability=1.0,
        minimum_visible_per_entity=0,
        minimum_visible_per_graph=1,
    )(batch, torch.Generator().manual_seed(0))
    assert torch.equal(masked.node_corrupted[0], batch.node_available[0])
    assert int((batch.node_available & ~masked.node_corrupted).sum()) == 1
