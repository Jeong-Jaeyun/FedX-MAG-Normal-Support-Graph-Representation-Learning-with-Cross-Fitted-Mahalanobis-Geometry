"""Chronology, circuit sequence, independent residual support and replay checks."""

import numpy as np
import pytest
import torch

from fedxmag_next.geometry import ConditionalMahalanobis
from fedxmag_next.h1_classical_features import SourcePCA
from fedxmag_next.h1_experiment import support_values, units
from fedxmag_next.h3_data import chronological_roles, histories, summarize
from fedxmag_next.h3_experiment import H3Detector, make_examples, post_attack_metrics, represented
from fedxmag_next.h3_models import SequentialMap, fit_gru, predict_gru, ridge_fit, ridge_predict


def test_role_examples_include_context_and_never_cross_boundaries_or_gaps():
    parents, examples = chronological_roles(np.r_[np.arange(180), np.arange(185, 400)])
    used = {}
    for role, minutes in examples.items():
        used[role] = set((minutes[:, None] - np.arange(14)).ravel())
        assert used[role] <= set(parents[role])
        assert min(minutes) >= min(parents[role]) + 13
    for a, role in enumerate(used):
        for other in list(used)[a + 1 :]:
            assert not used[role] & used[other]
    assert not set(range(180, 185)) & set.union(*used.values())


def test_sequence_quotas_handle_sparse_normal_minutes_without_empty_roles():
    # Complete normal runs are concentrated early; many isolated late normals
    # would make parent-count quantiles allocate no residual/calibration data.
    normal = np.r_[np.arange(330), np.arange(1000, 1900, 3)]
    parents, examples = chronological_roles(normal)
    assert all(len(v) >= 32 for v in examples.values())
    used = [set((v[:, None] - np.arange(14)).ravel()) for v in examples.values()]
    for i, a in enumerate(used):
        for b in used[i + 1 :]:
            assert not a & b
    assert all(
        set((v[:, None] - np.arange(14)).ravel()) <= set(parents[k]) for k, v in examples.items()
    )


def test_histories_use_elapsed_minutes_and_exclude_current_future():
    minutes = np.array([0, 1, 3, 4, 5])
    z = np.arange(40).reshape(5, 8)
    available = np.ones(5, bool)
    x, mask = histories([4], minutes, z, available, history=4)
    np.testing.assert_array_equal(mask, [[1, 1, 0, 1]])
    np.testing.assert_array_equal(x[0, 2], np.zeros(8))
    changed = z.copy()
    changed[minutes >= 4] = 9999
    np.testing.assert_array_equal(x, histories([4], minutes, changed, available, history=4)[0])
    with pytest.raises(ValueError):
        histories([4], [1, 1], np.zeros((2, 8)), [True, True])


def test_window_summary_is_permutation_invariant_within_minute():
    features = np.array([[1, 4], [3, 2], [2, 9]], dtype=float)
    times = np.array([62, 61, 180])
    minutes, values = summarize(features, times)
    permutation = np.array([2, 0, 1])
    other_minutes, other_values = summarize(features[permutation], times[permutation])
    np.testing.assert_array_equal(minutes, other_minutes)
    np.testing.assert_allclose(values, other_values)
    np.testing.assert_array_equal(values[0, :4], [2, 3, 1, 1])


@pytest.mark.parametrize("entangled", [False, True])
def test_temporal_circuit_matches_unfused_rotation_sequence(entangled):
    mapping = SequentialMap("quantum" if entangled else "quantum_noent", seed=29)
    x = np.random.default_rng(41).normal(size=(3, 8, 8))
    mask = np.ones((3, 8))
    mask[1, 3] = 0
    actual = mapping.transform(x, mask)[:, :32]
    state = np.zeros((3, 256), complex)
    state[:, 0] = 1
    circuit = mapping.circuit
    for t in range(8):
        angles = 2 * np.arctan(x[:, t] * mask[:, t, None]) + (1 - mask[:, t, None]) * np.pi / 2
        for j in range(8):
            circuit._gate(state, j, angles[:, j], "y")
            circuit._gate(state, j, angles[:, (j + 1) % 8], "z")
            circuit._gate(state, j, circuit.angles[0, j, 0], "y")
            circuit._gate(state, j, circuit.angles[0, j, 1], "z")
        if entangled:
            state *= circuit.phase
    np.testing.assert_allclose(actual, circuit._expectations(state), atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose((np.abs(state) ** 2).sum(1), 1, atol=1e-13)


def test_ablation_removes_values_and_order_control_changes_order():
    rng = np.random.default_rng(17)
    x, mask = rng.normal(size=(4, 8, 8)), np.ones((4, 8))
    zero = SequentialMap("quantum_zero_history")
    np.testing.assert_array_equal(zero.transform(x, mask), zero.transform(x * 20, 1 - mask))
    shuffled = SequentialMap("quantum_shuffled")
    standard = SequentialMap("quantum")
    np.testing.assert_allclose(
        shuffled.transform(x, mask), shuffled.transform(x[:, ::-1], mask[:, ::-1]), atol=1e-13
    )
    assert not np.allclose(standard.transform(x, mask), shuffled.transform(x, mask))


def test_ridge_selected_only_from_normal_validation_and_serializable(tmp_path):
    rng = np.random.default_rng(17)
    x, sx = rng.normal(size=(80, 10)), rng.normal(size=(30, 10))
    weights = rng.normal(size=(10, 8))
    y, sy = x @ weights, sx @ weights
    state = ridge_fit(x, y, sx, sy)
    assert ((ridge_predict(state, sx) - sy) ** 2).mean() < 0.01
    file = tmp_path / "ridge.pt"
    torch.save(state, file)
    np.testing.assert_array_equal(
        ridge_predict(state, sx), ridge_predict(torch.load(file, weights_only=True), sx)
    )


def test_gru_checkpoint_roundtrip(tmp_path):
    rng = np.random.default_rng(17)
    x, mask = rng.normal(size=(40, 8, 8)), np.ones((40, 8))
    target = 0.2 * x[:, -1]
    state = fit_gru(x[:30], mask[:30], target[:30], x[30:], mask[30:], target[30:], 17, epochs=2)
    file = tmp_path / "gru.pt"
    torch.save(state, file)
    np.testing.assert_array_equal(
        predict_gru(state, x, mask), predict_gru(torch.load(file, weights_only=True), x, mask)
    )


def test_frozen_detector_future_mutation_and_disk_restore(tmp_path):
    rng = np.random.default_rng(17)
    minutes = np.arange(200)
    summaries = rng.normal(size=(200, 18))
    timeline = dict(minutes=minutes, summary=summaries, available=np.ones(200, bool))
    adapter = SourcePCA.fit(summaries[:50])
    training = make_examples(timeline, adapter, np.arange(8, 50))
    selection = make_examples(timeline, adapter, np.arange(58, 80))
    support_examples = make_examples(timeline, adapter, np.arange(88, 130))
    mapping = SequentialMap("ar_ridge")
    state = dict(kind="ar_ridge", seed=17, map=mapping.state_dict())
    state["predictor"] = ridge_fit(
        mapping.transform(training["x"], training["mask"]),
        training["y"],
        mapping.transform(selection["x"], selection["mask"]),
        selection["y"],
    )
    reference = represented(state, support_examples)
    state.update(
        feature_mean=reference.mean(0), feature_scale=reference.std(0).clamp_min(0.05), supports={}
    )
    normal = (reference - state["feature_mean"]) / state["feature_scale"]
    for mode in (1, 8):
        geometry = ConditionalMahalanobis.fit(normal, max_components=mode)
        state["supports"][str(mode)] = dict(
            geometry=geometry.state_dict(), **units(support_values(geometry, normal))
        )
    artifact = dict(format_version=1, adapter=adapter.state_dict(), seeds=[state])
    detector = H3Detector(artifact)
    expected = detector.score_timeline(minutes, summaries, timeline["available"], [150])
    changed = summaries.copy()
    changed[151:] = 1e6
    for mode, score in detector.score_timeline(
        minutes, changed, timeline["available"], [150]
    ).items():
        torch.testing.assert_close(score, expected[mode], rtol=0, atol=0)
    file = tmp_path / "detector.pt"
    torch.save(artifact, file)
    restored = H3Detector(torch.load(file, weights_only=True))
    for mode, score in restored.score_timeline(
        minutes, summaries, timeline["available"], [150]
    ).items():
        torch.testing.assert_close(score, expected[mode], rtol=0, atol=0)


def test_recovery_counts_consecutive_normal_minutes_and_censoring():
    flags = np.array([True, True, False, True, True])
    labels = np.array([1, 0, 0, 1, 0])
    result = post_attack_metrics(flags, labels, np.arange(5), np.array([0, 1, 1, 1, 1], bool))
    assert result["normal_with_attack_in_previous8min"]["false_alarms"] == 2
    runs = result["contiguous_attack_to_normal_runs"]
    assert runs[0]["first_clear_after_minutes"] == 1
    assert runs[1]["right_censored"]
