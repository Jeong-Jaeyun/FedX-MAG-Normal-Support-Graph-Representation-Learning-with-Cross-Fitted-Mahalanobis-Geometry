from argparse import Namespace

import numpy as np
import torch
from test_context_association import observed_protocols
from torch.nn import functional as F

from fedxmag_next.daily_detector import DailyDeepEnergyDetector
from fedxmag_next.geometry import ConditionalMahalanobis
from fedxmag_next.geometry_training import GeometryResidualEncoder, geometry_loss
from fedxmag_next.record_data import FEATURES
from fedxmag_next.record_prequential import build_source
from fedxmag_next.temporal_context import CONTEXT_FEATURES


def test_geometry_loss_matches_frozen_normal_support_and_has_covariance_gradients():
    torch.manual_seed(38)
    model = GeometryResidualEncoder(5, hidden=7)
    values = torch.randn(192, 5)
    embedded = model(values)
    anchor, positive, negative = embedded.split((128, 32, 32))
    anchor.retain_grad()
    loss, real, changed = geometry_loss(anchor, positive, negative, 5)

    with torch.no_grad():
        center = anchor[:, 5:].mean(0)
        scale = anchor[:, 5:].std(0, unbiased=False).clamp_min(1e-6)

        def normalize(x):
            return torch.cat((x[:, :5], (x[:, 5:] - center) / scale), 1)

        support = ConditionalMahalanobis.fit(normalize(anchor), max_components=1)
        pos = support.score(normalize(positive))[0] / 12
        neg = support.score(normalize(negative))[0] / 12
        expected = F.softplus(1 + pos - neg).mean() + 0.1 * pos.mean()
        expected += 0.1 * F.relu(0.1 - anchor[:, 5:].std(0, unbiased=False)).mean()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(real, pos.mean())
        torch.testing.assert_close(changed, neg.mean())
    loss.backward()
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()
    assert anchor.grad.abs().sum() > 0
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_residual_encoder_keeps_raw_coordinates_and_query_batch_independence():
    torch.manual_seed(81)
    model = GeometryResidualEncoder(11).eval()
    query = torch.randn(40, 11)
    query[:, 3] = 0
    values = model.layers[:-1](query)
    assert torch.equal(values[:, :11], query)
    torch.testing.assert_close(values[:3], model(query[:3]))
    assert values.shape == (40, 75)


def test_geometry_source_restores_passthrough_scaling_and_support_energies(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(137)
    names = FEATURES + CONTEXT_FEATURES
    features = rng.normal(size=(256, len(names))).astype(np.float32)
    observed_protocols(features)
    features[:, 0] = 0
    data = {
        "features": features,
        "feature_names": np.array(names),
        "observed_at": np.arange(256),
        "signature": np.arange(256, dtype=np.uint64),
    }
    state = build_source(
        data,
        np.arange(192),
        np.arange(192, 256),
        Namespace(max_reference=192, seeds=[17], epochs=1, objective="geometry_margin"),
    )
    item = state["seed_models"][0]
    history = item["history"][0]
    assert history["anchor_probe_row_overlap"] == 0
    assert history["train_changed_context_pairs"] == 192
    assert torch.equal(item["embedding_mean"][: len(names)], torch.zeros(len(names)))
    assert torch.equal(item["embedding_scale"][: len(names)], torch.ones(len(names)))
    assert item["supports"]["1"]["geometry"]["means"].shape[1] == len(names) + 64
    path = tmp_path / "geometry.pt"
    torch.save(state, path)
    restored = DailyDeepEnergyDetector.load(path)
    query = torch.from_numpy(features[193:202])
    query[0, 0] = 1  # The raw source-constant coordinate still reaches the detector.
    normalized = restored.normalize_features(query)
    embedding = restored.models[0](normalized)
    assert torch.equal(embedding[:, : len(names)], normalized)
    embedding = (embedding - item["embedding_mean"]) / item["embedding_scale"]
    actual = restored.score_all(query)
    for modes in (1, 8):
        info = item["supports"][str(modes)]
        support = ConditionalMahalanobis.from_state_dict(info["geometry"])
        expected = (support.score(embedding)[0] - info["score_center"]) / info["score_scale"] + 1
        torch.testing.assert_close(actual[modes]["deep_mahalanobis"], expected)
        assert all(torch.isfinite(value).all() for value in actual[modes].values())
