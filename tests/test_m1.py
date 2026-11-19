import math

import numpy as np
import pytest
import torch

from fedxmag.losses import FAMILY_KEYS
from fedxmag.m1 import (
    ESTIMATORS,
    NormalSupportSettings,
    clean_view,
    embed_clean,
    estimate_covariance,
    ledoit_wolf_shrinkage,
    mahalanobis_energy,
    normal_support_loss,
    oas_shrinkage,
    train_m1,
)
from fedxmag.m1.detection import (
    MahalanobisDetector,
    auroc,
    average_precision,
    conformal_threshold,
    descriptor_matrix,
    structural_descriptors,
    threshold_metrics,
)
from fedxmag.masking import FamilyMasker
from fedxmag.models import M0GraphModel
from fedxmag.schema import DEFAULT_FEATURE_GROUPS, GraphSample, collate_graphs
from fedxmag.synthetic import make_synthetic_graphs


def _gaussian(samples: int, scales: list[float], seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((samples, len(scales)), generator=generator, dtype=torch.float64) * (
        torch.tensor(scales, dtype=torch.float64)
    )


def _ledoit_wolf_reference(values: torch.Tensor) -> float:
    """Direct Ledoit-Wolf (2004) estimator: alpha = min(b^2, d^2) / d^2."""

    centered = values - values.mean(dim=0)
    samples, features = centered.shape
    covariance = centered.T @ centered / samples
    mu = torch.trace(covariance) / features
    identity = torch.eye(features, dtype=torch.float64)
    d2 = (covariance - mu * identity).square().sum() / features
    b2_bar = sum(
        (torch.outer(row, row) - covariance).square().sum() / features for row in centered
    ) / samples**2
    return float(torch.minimum(b2_bar, d2) / d2)


def test_ledoit_wolf_matches_direct_paper_formula() -> None:
    for seed, scales in ((1, [3.0, 1.0, 0.5, 0.1]), (2, [1.0] * 6), (3, [5.0, 0.2])):
        values = _gaussian(40, scales, seed)
        centered = values - values.mean(dim=0)
        assert float(ledoit_wolf_shrinkage(centered)) == pytest.approx(
            _ledoit_wolf_reference(values), rel=1e-10, abs=1e-12
        )


def test_shrinkage_is_high_for_isotropic_and_low_for_anisotropic_data() -> None:
    isotropic = _gaussian(2000, [1.0] * 16, 4)
    anisotropic = _gaussian(2000, [10.0 ** (-k / 4) for k in range(16)], 5)
    for function in (ledoit_wolf_shrinkage, oas_shrinkage):
        assert float(function(isotropic - isotropic.mean(0))) > 0.8
        assert float(function(anisotropic - anisotropic.mean(0))) < 0.05


def test_in_sample_empirical_energy_is_the_algebraic_constant() -> None:
    values = _gaussian(50, [2.0, 1.0, 0.5, 0.25, 3.0], 6)
    estimate = estimate_covariance(values, "empirical", spectral_floor=0.0)
    mean_energy = float(mahalanobis_energy(values, estimate).mean())
    assert mean_energy == pytest.approx((50 - 1) * 5 / 50, rel=1e-10)
    shrunk = estimate_covariance(values, "ledoit_wolf", spectral_floor=0.0)
    assert float(mahalanobis_energy(values, shrunk).mean()) != pytest.approx(
        (50 - 1) * 5 / 50, rel=1e-6
    )


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_energy_is_scale_and_rotation_invariant(estimator: str) -> None:
    reference = _gaussian(60, [2.0, 1.0, 0.7, 0.3], 7)
    queries = _gaussian(9, [1.0, 1.0, 1.0, 1.0], 8)
    base = mahalanobis_energy(queries, estimate_covariance(reference, estimator))
    scaled = mahalanobis_energy(3.7 * queries, estimate_covariance(3.7 * reference, estimator))
    rotation, _ = torch.linalg.qr(_gaussian(4, [1.0] * 4, 9))
    rotated = mahalanobis_energy(
        queries @ rotation.T, estimate_covariance(reference @ rotation.T, estimator)
    )
    assert torch.allclose(base, scaled, rtol=1e-8)
    assert torch.allclose(base, rotated, rtol=1e-8)


def test_cross_fitted_loss_uses_disjoint_halves_and_stops_alpha_gradient() -> None:
    embeddings = _gaussian(32, [1.0] * 8, 10).float().requires_grad_(True)
    settings = NormalSupportSettings(estimator="ledoit_wolf", rank_min=4.0)
    loss = normal_support_loss(embeddings, settings, torch.Generator().manual_seed(1))
    loss.total.backward()
    assert embeddings.grad is not None and torch.isfinite(embeddings.grad).all()
    estimate = estimate_covariance(embeddings, "ledoit_wolf")
    assert not estimate.alpha.requires_grad
    in_sample = normal_support_loss(
        embeddings.detach(),
        NormalSupportSettings(estimator="empirical", energy_mode="in_sample", spectral_floor=0.0),
        torch.Generator().manual_seed(1),
    )
    assert float(in_sample.energy) == pytest.approx((32 - 1) / 32, rel=1e-6)


def test_shape_regularizer_penalizes_rank_collapse() -> None:
    healthy = _gaussian(64, [1.0] * 16, 11)
    collapsed = _gaussian(64, [1.0] + [1e-3] * 15, 12)
    settings = NormalSupportSettings(estimator="fixed", fixed_alpha=0.0, rank_min=8.0)
    generator = torch.Generator().manual_seed(2)
    healthy_loss = normal_support_loss(healthy, settings, generator)
    collapsed_loss = normal_support_loss(collapsed, settings, torch.Generator().manual_seed(2))
    assert float(healthy_loss.shape) == 0.0
    assert float(collapsed_loss.shape) > 1.0


def test_clean_view_has_no_artificial_corruption() -> None:
    batch = collate_graphs(make_synthetic_graphs(num_graphs=3, seed=13))
    view = clean_view(batch)
    assert not view.node_corrupted.any() and not view.edge_corrupted.any()
    assert not view.node_targeted.any() and not view.edge_targeted.any()
    assert torch.equal(view.node_visible, batch.node_features * batch.node_available)


@pytest.mark.parametrize(
    ("settings", "selection_mode"),
    [
        (NormalSupportSettings(lambda_m=0.0, lambda_c=0.0, lambda_s=0.0), "reconstruction"),
        (NormalSupportSettings(estimator="oas", rank_min=2.0), "reconstruction"),
        (
            NormalSupportSettings(reconstruction_weight=0.0, rank_min=2.0),
            "normal_support",
        ),
    ],
)
def test_m1_training_loop_records_geometry(
    settings: NormalSupportSettings, selection_mode: str
) -> None:
    groups = DEFAULT_FEATURE_GROUPS
    samples = make_synthetic_graphs(num_graphs=20, seed=17)
    model = M0GraphModel(
        groups.node_dim, groups.edge_dim, hidden_dim=12, graph_dim=4, num_layers=1, dropout=0.0
    )
    history = train_m1(
        model,
        samples[:16],
        samples[16:],
        FamilyMasker(0.35, 0.8),
        settings,
        epochs=2,
        batch_size=8,
        validation_batch_size=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clip_norm=5.0,
        early_stopping_patience=5,
        validation_mask_seeds=[3, 5],
        selection_mode=selection_mode,
        loss_weights={key: 1.0 for key in FAMILY_KEYS},
        seed=3,
        groups=groups,
    )
    assert len(history) == 2
    assert math.isfinite(history[-1]["selection_metric_value"])
    assert history[-1]["validation"]["clean_select_geometry"]["trace"] > 0
    if settings.uses_normal_support:
        assert "alpha" in history[-1]["training"]
    assert embed_clean(model, samples, groups).shape == (20, 4)


def test_ranking_metrics_match_brute_force() -> None:
    rng = np.random.default_rng(0)
    normal = np.round(rng.normal(0.0, 1.0, 60), 1)
    attack = np.round(rng.normal(1.0, 1.0, 40), 1)
    pairs = [(a > n) + 0.5 * (a == n) for a in attack for n in normal]
    assert auroc(normal, attack) == pytest.approx(np.mean(pairs))
    assert auroc([0, 1], [2, 3]) == 1.0 and auroc([2, 3], [0, 1]) == 0.0
    scores = np.concatenate((normal, attack))
    labels = np.concatenate((np.zeros(60), np.ones(40)))
    expected = 0.0
    previous_recall = 0.0
    for threshold in np.unique(scores)[::-1]:
        flagged = scores >= threshold
        recall = labels[flagged].sum() / 40
        expected += (recall - previous_recall) * labels[flagged].mean()
        previous_recall = recall
    assert average_precision(normal, attack) == pytest.approx(expected)


def test_conformal_threshold_and_threshold_metrics() -> None:
    calibration = np.arange(1, 101, dtype=float)
    assert conformal_threshold(calibration, 0.95) == 96.0
    assert conformal_threshold(calibration, 0.999) == 100.0
    metrics = threshold_metrics([1, 2, 3, 10], [5, 11, 12, 2], threshold=4.0)
    assert metrics["tpr"] == 0.75 and metrics["fpr"] == 0.25
    assert metrics["precision"] == 0.75


def test_structural_descriptors_on_a_known_graph() -> None:
    groups = DEFAULT_FEATURE_GROUPS
    sample = GraphSample(
        node_features=torch.zeros((3, groups.node_dim)),
        edge_features=torch.zeros((3, groups.edge_dim)),
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]]),
        node_available=torch.zeros((3, groups.node_dim), dtype=torch.bool),
        edge_available=torch.zeros((3, groups.edge_dim), dtype=torch.bool),
        metadata={"event_count": 9, "timing_observed_event_count": 6,
                  "synthetic_reverse_event_count": 3},
    )
    values = dict(
        zip(
            (
                "nodes", "edges", "density", "records", "timing", "reverse", "per_edge",
                "mean_out", "std_out", "max_out", "std_in", "max_in", "reciprocity",
                "source_only", "sink_only", "components", "largest",
            ),
            structural_descriptors(sample),
            strict=True,
        )
    )
    assert values["density"] == pytest.approx(0.5)
    assert values["reciprocity"] == pytest.approx(2 / 3)
    assert values["sink_only"] == pytest.approx(1 / 3)
    assert values["components"] == pytest.approx(1 / 3)
    assert values["reverse"] == pytest.approx(1 / 3)
    matrix = descriptor_matrix([sample, sample], "combined", groups)
    assert matrix.shape == (2, 17 + 3 * (groups.node_dim + groups.edge_dim))
    detector = MahalanobisDetector.fit(
        torch.randn(50, 5, dtype=torch.float64), standardize=True
    )
    assert detector.score(torch.zeros(1, 5, dtype=torch.float64)).shape == (1,)
