from argparse import Namespace

import numpy as np
import torch

from fedxmag_next.daily_detector import DailyDeepEnergyDetector
from fedxmag_next.deep_neighbors import DeepNeighborDetector, fit_neighbor_state, whiten
from fedxmag_next.geometry import ConditionalMahalanobis
from fedxmag_next.record_data import FEATURES
from fedxmag_next.record_prequential import build_source


def test_whitened_point_distance_equals_normal_covariance_mahalanobis_distance():
    torch.manual_seed(21)
    support = ConditionalMahalanobis.fit(torch.randn(128, 4), max_components=1)
    x, y = torch.randn(5, 4).double(), torch.randn(5, 4).double()
    actual = (whiten(x, support.state_dict()) - whiten(y, support.state_dict())).square().sum(1)
    covariance = support.cholesky[0] @ support.cholesky[0].T
    expected = ((x - y) * torch.linalg.solve(covariance, (x - y).T).T).sum(1)
    torch.testing.assert_close(actual, expected)


def test_deep_neighbor_fit_preserves_weights_and_restores_scores_including_empty_queries(tmp_path):
    torch.set_num_threads(2)
    features = np.random.default_rng(137).normal(size=(192, len(FEATURES))).astype(np.float32)
    data = {
        "features": features,
        "feature_names": np.array(FEATURES),
        "signature": np.arange(192, dtype=np.uint64),
        "observed_at": np.arange(192),
    }
    base = build_source(
        data,
        np.arange(128),
        np.arange(128, 192),
        Namespace(max_reference=128, seeds=[17], epochs=1),
    )
    state = fit_neighbor_state(base, torch.from_numpy(features[128:]))
    for name, value in base["seed_models"][0]["model"].items():
        assert torch.equal(value, state["base_encoder"]["seed_models"][0]["model"][name])
    path = tmp_path / "neighbors.pt"
    torch.save(state, path)
    original = DeepNeighborDetector(state)
    restored = DeepNeighborDetector.load(path)
    query = torch.from_numpy(features[131:140])
    scores, replay = original.score(query), restored.score(query)
    assert set(scores) == {
        "raw_whitened_knn",
        "deep_euclidean_knn_ensemble",
        "deep_whitened_knn_ensemble",
    }
    for key in scores:
        assert torch.isfinite(scores[key]).all()
        torch.testing.assert_close(scores[key], replay[key], rtol=0, atol=0)
        torch.testing.assert_close(scores[key][:2], restored.score(query[:2])[key])
    assert all(value.numel() == 0 for value in restored.score(query[:0]).values())
    assert all(
        parameter.dtype == torch.float32
        for parameter in DailyDeepEnergyDetector(base).models[0].parameters()
    )
