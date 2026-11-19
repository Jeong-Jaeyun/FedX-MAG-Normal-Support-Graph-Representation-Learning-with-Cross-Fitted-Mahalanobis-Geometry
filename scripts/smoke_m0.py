"""Run all M0 predictors on deterministic synthetic graphs."""

from __future__ import annotations

import json

import torch

from fedxmag.masking import FamilyMasker
from fedxmag.models import (
    AdjacencyShuffledGraphModel,
    ConstantPredictor,
    LocalPredictor,
    M0GraphModel,
    SetContextPredictor,
)
from fedxmag.schema import DEFAULT_FEATURE_GROUPS
from fedxmag.synthetic import make_synthetic_graphs
from fedxmag.training import evaluate_m0, train_m0


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(31)
    groups = DEFAULT_FEATURE_GROUPS
    samples = make_synthetic_graphs(num_graphs=32, seed=29)
    train_samples = samples[:24]
    validation_samples = samples[24:]
    masker = FamilyMasker(
        attribute_budget=0.30,
        family_mask_probability=0.50,
        minimum_visible_per_entity=0,
        minimum_visible_per_graph=1,
    )

    mean_model = ConstantPredictor()
    local_model = LocalPredictor(groups.node_dim, groups.edge_dim, hidden_dim=48, dropout=0.0)
    graph_model = M0GraphModel(
        groups.node_dim,
        groups.edge_dim,
        hidden_dim=48,
        graph_dim=24,
        num_layers=2,
        dropout=0.0,
    )
    set_context_model = SetContextPredictor(
        groups.node_dim,
        groups.edge_dim,
        hidden_dim=48,
        graph_dim=24,
        num_layers=2,
        dropout=0.0,
    )
    shuffled_graph_model = AdjacencyShuffledGraphModel(
        groups.node_dim,
        groups.edge_dim,
        hidden_dim=48,
        graph_dim=24,
        num_layers=2,
        dropout=0.0,
    )

    train_m0(
        local_model,
        train_samples,
        validation_samples,
        masker,
        epochs=5,
        batch_size=8,
        seed=31,
    )
    train_m0(
        set_context_model,
        train_samples,
        validation_samples,
        masker,
        epochs=5,
        batch_size=8,
        seed=31,
    )
    train_m0(
        shuffled_graph_model,
        train_samples,
        validation_samples,
        masker,
        epochs=5,
        batch_size=8,
        seed=31,
    )
    train_m0(
        graph_model,
        train_samples,
        validation_samples,
        masker,
        epochs=5,
        batch_size=8,
        seed=31,
    )
    evaluation_seed = 123_457
    report = {
        "warning": "Synthetic smoke results are software checks, not scientific evidence.",
        "mean": evaluate_m0(
            mean_model,
            validation_samples,
            masker,
            batch_size=8,
            seed=evaluation_seed,
        ),
        "local": evaluate_m0(
            local_model,
            validation_samples,
            masker,
            batch_size=8,
            seed=evaluation_seed,
        ),
        "set_context": evaluate_m0(
            set_context_model,
            validation_samples,
            masker,
            batch_size=8,
            seed=evaluation_seed,
        ),
        "shuffled_graph": evaluate_m0(
            shuffled_graph_model,
            validation_samples,
            masker,
            batch_size=8,
            seed=evaluation_seed,
        ),
        "graph": evaluate_m0(
            graph_model,
            validation_samples,
            masker,
            batch_size=8,
            seed=evaluation_seed,
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
