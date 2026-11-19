import math

import numpy as np
import pytest
import torch

from fedxmag.losses import FAMILY_KEYS
from fedxmag.m1.entity import (
    EntityScorer,
    EntitySupportSettings,
    StateGraphModel,
    aggregate,
    encode_entities,
    entity_support_loss,
    raw_entity_features,
    train_entity_m1,
)
from fedxmag.m1.training import clean_view
from fedxmag.masking import FamilyMasker
from fedxmag.models import M0GraphModel
from fedxmag.schema import DEFAULT_FEATURE_GROUPS, collate_graphs
from fedxmag.synthetic import make_synthetic_graphs

KWARGS = {"node_dim": DEFAULT_FEATURE_GROUPS.node_dim, "edge_dim": DEFAULT_FEATURE_GROUPS.edge_dim,
          "hidden_dim": 12, "graph_dim": 4, "num_layers": 2, "dropout": 0.0}


def test_state_model_is_a_drop_in_m0_graph_model() -> None:
    torch.manual_seed(0)
    reference = M0GraphModel(**KWARGS).eval()
    model = StateGraphModel(**KWARGS).eval()
    model.load_state_dict(reference.state_dict())
    batch = collate_graphs(make_synthetic_graphs(num_graphs=3, seed=3))
    masked = FamilyMasker(0.35, 0.8)(batch, torch.Generator().manual_seed(1))
    output, node_state, edge_state = model.forward_with_states(masked)
    expected = reference(masked)
    assert torch.equal(output.node_prediction, expected.node_prediction)
    assert torch.equal(output.graph_embedding, expected.graph_embedding)
    assert node_state.shape == (batch.node_features.shape[0], 12)
    assert edge_state.shape == (batch.edge_features.shape[0], 12)


def test_entity_scores_and_aggregation() -> None:
    samples = make_synthetic_graphs(num_graphs=6, seed=5)
    model = StateGraphModel(**KWARGS)
    states = encode_entities(model, samples, DEFAULT_FEATURE_GROUPS, batch_size=4)
    assert len(states.node) == 6 and states.node[0].dtype == torch.float64
    scorer = EntityScorer.fit(states, "node")
    graph = scorer.graph_scores(states, "top3_mean")
    assert graph.shape == (6,) and np.isfinite(graph).all()
    raw = raw_entity_features(samples)
    assert raw.edge[0].shape[1] == 2 * DEFAULT_FEATURE_GROUPS.edge_dim
    values = [torch.tensor([1.0, 5.0, 3.0, 2.0]), torch.tensor([]), torch.tensor([4.0])]
    assert aggregate(values, "max").tolist() == [5.0, 0.0, 4.0]
    assert aggregate(values, "top3_mean")[0] == pytest.approx(10 / 3)
    assert aggregate(values, "mean")[0] == pytest.approx(2.75)


def test_entity_support_loss_uses_graph_disjoint_halves_and_backpropagates() -> None:
    batch = collate_graphs(make_synthetic_graphs(num_graphs=8, seed=7))
    masked = clean_view(batch)
    model = StateGraphModel(**KWARGS)
    _, node_state, edge_state = model.forward_with_states(masked)
    settings = EntitySupportSettings(rank_min=2.0)
    loss, diagnostics = entity_support_loss(
        {"node": node_state, "edge": edge_state},
        {"node": batch.node_graph_index, "edge": batch.edge_graph_index},
        batch.num_graphs, settings, torch.Generator().manual_seed(2),
    )
    loss.backward()
    assert math.isfinite(float(loss.detach())) and diagnostics["energy_over_d"] > 0
    assert all(p.grad is not None for p in model.node_input.parameters())
    order = torch.randperm(8, generator=torch.Generator().manual_seed(2))
    assert set(order[:4].tolist()).isdisjoint(order[4:].tolist())


def test_rank_guardrail_penalizes_collapsed_states() -> None:
    graph_index = torch.arange(8).repeat_interleave(10)
    healthy = torch.randn(80, 16, generator=torch.Generator().manual_seed(3))
    collapsed = torch.randn(80, 1, generator=torch.Generator().manual_seed(3)) @ torch.ones(1, 16)
    collapsed = collapsed + 1e-3 * healthy
    settings = EntitySupportSettings(kinds=("node",), rank_min=8.0)
    _, good = entity_support_loss({"node": healthy}, {"node": graph_index}, 8, settings,
                                  torch.Generator().manual_seed(4))
    _, bad = entity_support_loss({"node": collapsed}, {"node": graph_index}, 8, settings,
                                 torch.Generator().manual_seed(4))
    assert good["rank_penalty"] == 0.0 and bad["rank_penalty"] > 1.0


def test_control_entity_states_have_entity_shapes() -> None:
    from fedxmag.m1.entity import encode_control_entities
    from fedxmag.models import AdjacencyShuffledGraphModel, SetContextPredictor

    samples = make_synthetic_graphs(num_graphs=5, seed=13)
    for model in (SetContextPredictor(**KWARGS), AdjacencyShuffledGraphModel(**KWARGS)):
        states = encode_control_entities(model, samples, DEFAULT_FEATURE_GROUPS, batch_size=2)
        assert [s.shape[0] for s in states.node] == [s.node_features.shape[0] for s in samples]
        assert [s.shape[0] for s in states.edge] == [s.edge_features.shape[0] for s in samples]


def test_stratified_aurocs_and_localization() -> None:
    from fedxmag.m1.entity import localization_rate, stratified_aurocs

    rng = np.random.default_rng(0)
    normal, attack = rng.normal(0, 1, 40), rng.normal(1, 1, 60)
    result = stratified_aurocs(
        normal, attack,
        rng.integers(2, 20, 40).astype(float), rng.integers(2, 20, 60).astype(float),
        np.asarray(["d1", "d2"] * 20), np.asarray(["d1", "d2", "d3"] * 20),
        rng.uniform(0, 1, 60),
    )
    assert result["shared_dates"] == 2.0 and 0.5 < result["within_date_auroc"] <= 1.0
    assert result["size_stratified_auroc"] is not None
    rate = localization_rate(
        [torch.tensor([0.1, 5.0, 0.2]), torch.tensor([3.0, 0.1])],
        [["a", "b", "c"], ["x", "y"]],
        [{"b"}, {"y"}],
    )
    assert rate["top1_is_attack_endpoint"] == 0.5 and rate["windows"] == 2.0


def test_entity_m1_training_runs() -> None:
    samples = make_synthetic_graphs(num_graphs=20, seed=11)
    model = StateGraphModel(**KWARGS)
    history = train_entity_m1(
        model, samples[:16], samples[16:], FamilyMasker(0.35, 0.8),
        EntitySupportSettings(rank_min=2.0), epochs=2, batch_size=8, validation_batch_size=4,
        learning_rate=1e-3, weight_decay=0.0, gradient_clip_norm=5.0,
        early_stopping_patience=5, validation_mask_seeds=[3, 5],
        loss_weights={key: 1.0 for key in FAMILY_KEYS}, seed=3, groups=DEFAULT_FEATURE_GROUPS,
    )
    assert len(history) == 2
    assert "node_alpha" in history[-1]["training"]
    assert history[-1]["validation"]["clean_select_node_effective_rank"] > 0
