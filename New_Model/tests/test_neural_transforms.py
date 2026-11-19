import math
from argparse import Namespace

import numpy as np
import torch

from fedxmag_next.daily_detector import DailyDeepEnergyDetector
from fedxmag_next.neural_transforms import TransformationEnergy, deterministic_contrastive_score
from fedxmag_next.record_data import FEATURES
from fedxmag_next.record_prequential import build_source


def test_dcl_denominator_excludes_self_and_is_invariant_to_view_order():
    # Identical views yield K*log(K), not K*log(K+1).
    constant = torch.ones(3, 5, 7)
    expected = torch.full((3,), 4 * math.log(4))
    torch.testing.assert_close(deterministic_contrastive_score(constant, 0.1), expected)
    generator = torch.Generator().manual_seed(71)
    states = torch.randn(5, 5, 9, generator=generator)
    original = deterministic_contrastive_score(states, 0.1)
    permuted = deterministic_contrastive_score(states[:, [0, 3, 1, 4, 2]], 0.1)
    torch.testing.assert_close(original, permuted)
    # Direct definition from Eq.2, using independent scalar operations.
    z = torch.nn.functional.normalize(states.double(), dim=-1)
    scalar = []
    for sample in z:
        value = 0.0
        for k in range(1, len(sample)):
            positive = torch.exp(torch.dot(sample[k], sample[0]) / 0.1)
            denominator = positive + sum(
                torch.exp(torch.dot(sample[k], sample[j]) / 0.1)
                for j in range(1, len(sample))
                if j != k
            )
            value -= torch.log(positive / denominator)
        scalar.append(value)
    torch.testing.assert_close(original.double(), torch.stack(scalar), atol=2e-5, rtol=1e-6)


def test_scores_do_not_depend_on_other_query_rows_and_gradients_reach_both_components():
    torch.manual_seed(17)
    model = TransformationEnergy(11)
    values = torch.randn(7, 11)
    batch = model(values)
    alone = torch.cat([model(row[None]) for row in values])
    torch.testing.assert_close(batch, alone, atol=2e-5, rtol=1e-6)
    batch.mean().backward()
    for group in (model.layers, model.masks):
        gradients = [p.grad for p in group.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0


def test_trained_source_artifact_restores_native_scoring_and_mahalanobis(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(71)
    features = rng.normal(size=(160, len(FEATURES))).astype(np.float32)
    data = {
        "features": features,
        "feature_names": np.array(FEATURES),
        "observed_at": np.arange(160),
        "signature": np.arange(160, dtype=np.uint64),
    }
    state = build_source(
        data,
        np.arange(128),
        np.arange(128, 160),
        Namespace(max_reference=128, seeds=[17], epochs=1, objective="neural_transformations"),
    )
    path = tmp_path / "source_detector.pt"
    torch.save(state, path)
    restored = DailyDeepEnergyDetector.load(path)
    query = torch.from_numpy(features[:9])
    values = restored.normalize_features(query)
    direct = TransformationEnergy(len(FEATURES))
    direct.load_state_dict(state["seed_models"][0]["model"])
    fitted = state["seed_models"][0]
    expected = (direct(values).double() - fitted["native_score_center"]) / fitted[
        "native_score_scale"
    ] + 1
    result = restored.score_all(query)
    for modes in (1, 8):
        torch.testing.assert_close(result[modes]["transformation"], expected)
        assert all(torch.isfinite(value).all() for value in result[modes].values())
    torch.testing.assert_close(direct.layers[:-1](values).norm(dim=1), torch.ones(9))
