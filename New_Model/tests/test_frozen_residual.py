from argparse import Namespace

import numpy as np
import torch

from fedxmag_next.daily_detector import DailyDeepEnergyDetector
from fedxmag_next.frozen_residual import add_raw_bypass
from fedxmag_next.record_data import FEATURES
from fedxmag_next.record_prequential import build_source


def test_frozen_bypass_preserves_neural_weights_raw_inputs_controls_and_restore(tmp_path):
    torch.set_num_threads(2)
    features = np.random.default_rng(109).normal(size=(192, len(FEATURES))).astype(np.float32)
    features[:, 3] = 0
    data = {
        "features": features,
        "feature_names": np.array(FEATURES),
        "signature": np.arange(192, dtype=np.uint64),
        "observed_at": np.arange(192),
    }
    state = build_source(
        data,
        np.arange(128),
        np.arange(128, 192),
        Namespace(max_reference=128, seeds=[17], epochs=1),
    )
    reference, selected = map(torch.from_numpy, (features[:128], features[128:]))
    result = add_raw_bypass(state, reference, selected)
    assert state["encoder_kind"] == "dependency"
    base, augmented = DailyDeepEnergyDetector(state), DailyDeepEnergyDetector(result)
    query = selected[:9].clone()
    query[0, 3] = 1
    normalized = base.normalize_features(query)
    with torch.no_grad():
        old = base.models[0].layers[:-1](normalized)
        new = augmented.models[0](normalized)
    assert torch.equal(new[:, : len(FEATURES)], normalized)
    torch.testing.assert_close(new[:, len(FEATURES) :], old, rtol=0, atol=0)
    assert (
        result["seed_models"][0]["supports"]["1"]["geometry"]["means"].shape[1]
        == len(FEATURES) + 64
    )
    path = tmp_path / "frozen.pt"
    torch.save(result, path)
    actual, replay = augmented.score_all(query), DailyDeepEnergyDetector.load(path).score_all(query)
    original_scores = base.score_all(query)
    for modes in (1, 8):
        for key in actual[modes]:
            assert torch.isfinite(actual[modes][key]).all()
            torch.testing.assert_close(actual[modes][key], replay[modes][key], rtol=0, atol=0)
        for key in ("raw_knn", "raw_mahalanobis"):
            torch.testing.assert_close(
                actual[modes][key], original_scores[modes][key], rtol=0, atol=0
            )
