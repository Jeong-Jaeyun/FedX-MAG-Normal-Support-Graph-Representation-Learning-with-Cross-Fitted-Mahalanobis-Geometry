import math

import pytest
import torch

from fedxmag.losses import FAMILY_KEYS, m0_loss
from fedxmag.masking import FamilyMasker, MaskedGraphBatch
from fedxmag.models import (
    AdjacencyShuffledGraphModel,
    ConstantPredictor,
    LocalPredictor,
    M0GraphModel,
    M0Output,
    SetContextPredictor,
)
from fedxmag.models.m0 import _shuffle_endpoints_within_graph
from fedxmag.schema import DEFAULT_FEATURE_GROUPS, GraphBatch, collate_graphs
from fedxmag.synthetic import make_synthetic_graphs
from fedxmag.training import evaluate_m0, train_m0


def test_graph_model_shapes_and_backward() -> None:
    groups = DEFAULT_FEATURE_GROUPS
    batch = collate_graphs(make_synthetic_graphs(num_graphs=3, seed=19))
    masked = FamilyMasker(0.40, 1.0)(batch, torch.Generator().manual_seed(2))
    model = M0GraphModel(
        groups.node_dim,
        groups.edge_dim,
        hidden_dim=16,
        graph_dim=8,
        num_layers=2,
        dropout=0.0,
    )
    output = model(masked)
    assert output.node_prediction.shape == batch.node_features.shape
    assert output.edge_prediction.shape == batch.edge_features.shape
    assert output.graph_embedding.shape == (3, 8)
    loss = m0_loss(output, masked)
    assert sum(loss.counts.values()) > 0
    loss.total.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_all_pretext_models_report_each_family() -> None:
    groups = DEFAULT_FEATURE_GROUPS
    samples = make_synthetic_graphs(num_graphs=8, seed=23)
    masker = FamilyMasker(0.35, 0.8)
    models = [
        ConstantPredictor(),
        LocalPredictor(groups.node_dim, groups.edge_dim, hidden_dim=12, dropout=0.0),
        SetContextPredictor(
            groups.node_dim,
            groups.edge_dim,
            hidden_dim=12,
            graph_dim=6,
            num_layers=1,
            dropout=0.0,
        ),
        AdjacencyShuffledGraphModel(
            groups.node_dim,
            groups.edge_dim,
            hidden_dim=12,
            graph_dim=6,
            num_layers=1,
            dropout=0.0,
        ),
        M0GraphModel(
            groups.node_dim,
            groups.edge_dim,
            hidden_dim=12,
            graph_dim=6,
            num_layers=1,
            dropout=0.0,
        ),
    ]
    for model in models:
        metrics = evaluate_m0(model, samples, masker, batch_size=4, seed=101)
        for key in FAMILY_KEYS:
            assert metrics[f"{key}_count"] > 0
            assert math.isfinite(metrics[f"{key}_mse"])
        assert metrics["degree_strata"]
        assert all(
            math.isfinite(item["mse"])
            for item in metrics["degree_strata"].values()
        )


def test_set_context_is_exactly_capacity_matched_and_adjacency_invariant() -> None:
    groups = DEFAULT_FEATURE_GROUPS
    samples = make_synthetic_graphs(num_graphs=3, seed=29)
    batch = collate_graphs(samples)
    masked = FamilyMasker(0.40, 1.0)(batch, torch.Generator().manual_seed(7))
    graph = M0GraphModel(
        groups.node_dim,
        groups.edge_dim,
        hidden_dim=16,
        graph_dim=8,
        num_layers=2,
        dropout=0.0,
    ).eval()
    set_context = SetContextPredictor(
        groups.node_dim,
        groups.edge_dim,
        hidden_dim=16,
        graph_dim=8,
        num_layers=2,
        dropout=0.0,
    ).eval()
    graph_parameters = sum(parameter.numel() for parameter in graph.parameters())
    set_parameters = sum(parameter.numel() for parameter in set_context.parameters())
    assert set_parameters == graph_parameters

    original = set_context(masked)
    swapped_batch = GraphBatch(
        **{
            **vars(batch),
            "edge_index": batch.edge_index.flip(0),
        }
    )
    swapped = set_context(
        MaskedGraphBatch(
            **{
                **vars(masked),
                "batch": swapped_batch,
            }
        )
    )
    assert torch.equal(original.node_prediction, swapped.node_prediction)
    assert torch.equal(original.edge_prediction, swapped.edge_prediction)


def test_shuffled_graph_control_is_capacity_matched() -> None:
    groups = DEFAULT_FEATURE_GROUPS
    kwargs = {
        "node_dim": groups.node_dim,
        "edge_dim": groups.edge_dim,
        "hidden_dim": 16,
        "graph_dim": 8,
        "num_layers": 2,
        "dropout": 0.0,
    }
    graph = M0GraphModel(**kwargs)
    shuffled = AdjacencyShuffledGraphModel(**kwargs)
    assert sum(parameter.numel() for parameter in graph.parameters()) == sum(
        parameter.numel() for parameter in shuffled.parameters()
    )


def test_both_endpoint_shuffle_preserves_degrees_and_breaks_both_associations() -> None:
    sample = make_synthetic_graphs(num_graphs=1, min_nodes=7, max_nodes=7, seed=31)[0]
    batch = collate_graphs([sample])
    masked = FamilyMasker(0.40, 1.0)(
        batch, torch.Generator().manual_seed(13)
    )
    shuffled = _shuffle_endpoints_within_graph(
        masked, seed=271_828, mode="both_endpoints"
    )
    source, destination = batch.edge_index
    shuffled_source, shuffled_destination = shuffled.batch.edge_index

    assert torch.equal(torch.bincount(source), torch.bincount(shuffled_source))
    assert torch.equal(
        torch.bincount(destination), torch.bincount(shuffled_destination)
    )
    assert not torch.equal(source, shuffled_source)
    assert not torch.equal(destination, shuffled_destination)
    assert torch.equal(batch.edge_features, shuffled.batch.edge_features)


def test_short_training_loop_runs() -> None:
    groups = DEFAULT_FEATURE_GROUPS
    samples = make_synthetic_graphs(num_graphs=8, seed=41)
    model = LocalPredictor(groups.node_dim, groups.edge_dim, hidden_dim=12, dropout=0.0)
    history = train_m0(
        model,
        samples[:6],
        samples[6:],
        FamilyMasker(0.35, 0.8),
        epochs=2,
        batch_size=3,
        seed=5,
    )
    assert len(history) == 2
    assert math.isfinite(history[-1]["validation"]["macro_family_mse"])
    assert history[-1]["selection_metric"] == (
        "validation_mask_mean_macro_family_mse"
    )
    assert history[-1]["selection_metric_value"] == (
        history[-1]["validation"]["macro_family_mse_mean"]
    )


def test_training_rejects_unimplemented_selection_metric() -> None:
    groups = DEFAULT_FEATURE_GROUPS
    samples = make_synthetic_graphs(num_graphs=4, seed=43)
    model = LocalPredictor(groups.node_dim, groups.edge_dim, hidden_dim=8, dropout=0.0)
    with pytest.raises(ValueError, match="Unsupported selection metric"):
        train_m0(
            model,
            samples[:2],
            samples[2:],
            FamilyMasker(0.35, 0.8),
            epochs=1,
            selection_metric="not_implemented",
        )


def test_loss_uses_graph_macro_not_scalar_pooling() -> None:
    groups = DEFAULT_FEATURE_GROUPS
    node_features = torch.zeros((11, groups.node_dim))
    node_features[0, 0] = 1.0
    node_available = torch.zeros_like(node_features, dtype=torch.bool)
    node_available[:, 0] = True
    edge_features = torch.empty((0, groups.edge_dim))
    edge_available = torch.empty((0, groups.edge_dim), dtype=torch.bool)
    batch = GraphBatch(
        node_features=node_features,
        edge_features=edge_features,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        node_available=node_available,
        edge_available=edge_available,
        node_graph_index=torch.tensor([0] + [1] * 10),
        edge_graph_index=torch.empty(0, dtype=torch.long),
        labels=torch.zeros(2, dtype=torch.long),
        num_graphs=2,
    )
    targeted = node_available.clone()
    masked = MaskedGraphBatch(
        batch=batch,
        node_visible=torch.zeros_like(node_features),
        edge_visible=edge_features,
        node_corrupted=targeted.clone(),
        edge_corrupted=edge_available.clone(),
        node_targeted=targeted,
        edge_targeted=edge_available.clone(),
    )
    output = M0Output(
        node_prediction=torch.zeros_like(node_features),
        edge_prediction=edge_features,
        graph_embedding=torch.empty((2, 0)),
    )
    loss = m0_loss(output, masked)
    assert torch.isclose(loss.components["node_timing"], torch.tensor(0.5))
    assert loss.graph_counts["node_timing"] == 2
