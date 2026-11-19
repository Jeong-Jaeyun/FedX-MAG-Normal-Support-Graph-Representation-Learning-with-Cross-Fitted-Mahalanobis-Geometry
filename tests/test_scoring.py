from dataclasses import replace

import numpy as np
import pytest
import torch

from fedxmag.m1.entity import EntityScorer, StateGraphModel, encode_entities
from fedxmag.schema import DEFAULT_FEATURE_GROUPS, RESOLUTION_AWARE_FEATURE_GROUPS
from fedxmag.scoring import FrozenNodeDetector
from fedxmag.synthetic import make_synthetic_graphs

GROUPS = DEFAULT_FEATURE_GROUPS
KWARGS = {
    "node_dim": GROUPS.node_dim,
    "edge_dim": GROUPS.edge_dim,
    "hidden_dim": 8,
    "graph_dim": 4,
    "num_layers": 1,
    "dropout": 0.1,
}


@pytest.fixture
def fitted() -> FrozenNodeDetector:
    torch.manual_seed(79)
    model = StateGraphModel(**KWARGS)
    train = make_synthetic_graphs(num_graphs=12, seed=71)
    calibration = make_synthetic_graphs(num_graphs=100, seed=73)
    return FrozenNodeDetector.fit(
        model, KWARGS, GROUPS, train, calibration, provenance={"seed": 79}
    )


def test_frozen_detector_matches_confirmed_scorer_and_survives_serialization(
    fitted: FrozenNodeDetector,
    tmp_path,
) -> None:
    samples = make_synthetic_graphs(num_graphs=5, seed=83)
    reference = encode_entities(fitted.model, make_synthetic_graphs(num_graphs=12, seed=71), GROUPS)
    scorer = EntityScorer.fit(reference, "node", "ledoit_wolf")
    expected = scorer.graph_scores(encode_entities(fitted.model, samples, GROUPS), "top3_mean")
    scores, _ = fitted.score(samples)
    np.testing.assert_array_equal(scores, expected)
    assert not fitted.model.training
    assert all(not parameter.requires_grad for parameter in fitted.model.parameters())
    path = tmp_path / "detector.pt"
    fitted.save(path)
    restored = FrozenNodeDetector.load(path)
    assert restored.thresholds == fitted.thresholds
    assert restored.provenance == {"seed": 79}
    np.testing.assert_array_equal(restored.score(samples)[0], scores)
    with pytest.raises(FileExistsError):
        fitted.save(path)


def test_predictions_localize_nodes_and_use_strict_thresholds(fitted: FrozenNodeDetector) -> None:
    samples = make_synthetic_graphs(num_graphs=1, min_nodes=2, max_nodes=2, seed=89)
    sample = samples[0]
    sample.metadata["endpoint_keys"] = ["host-a", "host-b"]
    score, energies = fitted.score(samples)
    # Equality to the threshold is not an alert, matching the registered scorer.
    fitted.calibration_scores[:] = score[0]
    prediction = fitted.predict(samples)[0]
    assert prediction["score"] == pytest.approx(float(energies[0].mean()))
    assert not any(prediction["alerts"].values())
    assert len(prediction["top_nodes"]) == 2
    highest = int(torch.argmax(energies[0]))
    assert prediction["top_nodes"][0]["endpoint_key"] == sample.metadata["endpoint_keys"][highest]
    # Prediction ignores labels and attack metadata.
    attack = replace(sample, label=1, metadata={**sample.metadata, "attack_categories": ["DoS"]})
    assert fitted.predict([attack]) == [prediction]
    sample.metadata["endpoint_keys"] = ["missing-node"]
    with pytest.raises(ValueError, match="endpoint_keys"):
        fitted.predict(samples)


@pytest.mark.parametrize("role", ("train", "calibration"))
def test_fitting_rejects_attack_data(role: str) -> None:
    train = make_synthetic_graphs(num_graphs=3, seed=97)
    calibration = make_synthetic_graphs(num_graphs=100, seed=101)
    (train if role == "train" else calibration)[0].label = 1
    with pytest.raises(ValueError, match="normal-only"):
        FrozenNodeDetector.fit(StateGraphModel(**KWARGS), KWARGS, GROUPS, train, calibration)


def test_rejects_missing_calibration_and_incompatible_artifacts(fitted, tmp_path) -> None:
    train = make_synthetic_graphs(num_graphs=3, seed=103)
    with pytest.raises(ValueError, match="normal-only"):
        FrozenNodeDetector.fit(StateGraphModel(**KWARGS), KWARGS, GROUPS, train, [])
    with pytest.raises(ValueError, match="99 calibration"):
        FrozenNodeDetector.fit(StateGraphModel(**KWARGS), KWARGS, GROUPS, train, train)
    with pytest.raises(ValueError, match="batch_size"):
        fitted.score(train, batch_size=0)
    assert fitted.predict([]) == []
    path = tmp_path / "detector.pt"
    fitted.save(path)
    payload = torch.load(path, weights_only=True)
    payload["groups"]["node_timing"] = RESOLUTION_AWARE_FEATURE_GROUPS.node_timing
    torch.save(payload, tmp_path / "bad_schema.pt")
    with pytest.raises(ValueError, match="schema"):
        FrozenNodeDetector.load(tmp_path / "bad_schema.pt")
    payload = torch.load(path, weights_only=True)
    payload["thresholds"]["0.05"] += 1.0
    torch.save(payload, tmp_path / "bad_threshold.pt")
    with pytest.raises(ValueError, match="thresholds"):
        FrozenNodeDetector.load(tmp_path / "bad_threshold.pt")
