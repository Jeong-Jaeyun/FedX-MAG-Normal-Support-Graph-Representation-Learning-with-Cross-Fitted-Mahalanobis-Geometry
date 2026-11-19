"""Generator invariants and scientific negative-control boundaries."""

from itertools import combinations, product

import numpy as np
import pytest
import torch

from fedxmag_next.h1_classical_features import SourcePCA
from fedxmag_next.h1_experiment import units
from fedxmag_next.h2_experiment import make_model, model_input
from fedxmag_next.h23_mechanisms import (
    DOMAINS,
    HISTORY,
    finite_metrics,
    fit_row_state,
    generic_corrupt,
    reverse_blocks,
    role_observations,
    row_draw,
    row_ensemble_scores,
    row_scores,
    row_tasks,
    sequence_examples,
    temporal_draw,
    temporal_roles,
    temporal_tasks,
)


def test_parity_population_lower_marginals_match_but_third_moment_changes():
    # Exact enumeration proves the sign mechanism, unlike finite-draw moment equality.
    signs = np.array(list(product((-1.0, 1.0), repeat=2)))
    normal = np.c_[signs, signs[:, 0] * signs[:, 1]]
    anomaly = normal.copy()
    anomaly[:, 2] *= -1
    for size in (1, 2):
        for cols in combinations(range(3), size):
            assert sorted(map(tuple, normal[:, cols])) == sorted(map(tuple, anomaly[:, cols]))
    assert np.mean(np.prod(normal, axis=1)) == 1
    assert np.mean(np.prod(anomaly, axis=1)) == -1


@pytest.mark.parametrize("config", DOMAINS)
def test_draws_preserve_masks_and_parity_claim_with_new_source_test_samples(config):
    d = config["dimension"]
    normal = row_draw(128, config, 101)
    flipped = row_draw(128, config, 101, "parity_flip")
    assert normal.shape == (128, 2 * d)
    np.testing.assert_array_equal(normal[:, d:], flipped[:, d:])
    np.testing.assert_array_equal(normal[:, 2], -flipped[:, 2])
    np.testing.assert_array_equal(
        normal[:, np.arange(2 * d) != 2], flipped[:, np.arange(2 * d) != 2]
    )
    assert np.isin(normal[:, d:], (0, 1)).all()
    assert (normal[:, :d][normal[:, d:] == 0] == 0).all()
    assert not np.array_equal(normal, row_draw(128, config, 102))


@pytest.mark.parametrize("policy", ("original", "conditional"))
def test_generic_corruption_has_only_normal_numeric_availability_donors(policy):
    config = DOMAINS[2]
    x = row_draw(100, config, 123)
    changed = generic_corrupt(x, policy, 17).numpy()
    np.testing.assert_array_equal(changed, generic_corrupt(x, policy, 17).numpy())
    d = config["dimension"]
    for j in range(d):
        donors = set(map(tuple, x[:, [j, d + j]]))
        assert all(tuple(v) in donors for v in changed[:, [j, d + j]])
    if policy == "conditional":
        np.testing.assert_array_equal(changed[:, d:], x[:, d:])
        np.testing.assert_array_equal(changed[:, d - 1], x[:, d - 1])


def test_source_roles_include_history_and_never_share_an_observation():
    roles, length = temporal_roles((64, 32, 32, 32))
    owned = role_observations(roles, HISTORY)
    assert min(owned["train"]) == 0
    assert max(owned["calibration"]) < length
    for a, b in combinations(owned, 2):
        assert not np.intersect1d(owned[a], owned[b]).size
    with pytest.raises(ValueError, match="overlap"):
        role_observations({"a": np.arange(8, 20), "b": np.arange(20, 30)}, HISTORY)


def test_rotation_process_is_nonreversible_and_reversal_preserves_exact_observed_bag():
    raw, transition = temporal_draw(96, DOMAINS[1], 173)
    assert not np.allclose(transition, transition.T)
    changed = reverse_blocks(raw, 32)
    np.testing.assert_array_equal(raw[:32], changed[:32])
    assert not np.array_equal(raw[32:], changed[32:])
    for start in range(32, len(raw), 16):
        assert sorted(map(tuple, raw[start : start + 16])) == sorted(
            map(tuple, changed[start : start + 16])
        )


def test_sequence_input_excludes_current_and_future_and_uses_frozen_adapter():
    raw, _ = temporal_draw(100, DOMAINS[0], 901)
    adapter = SourcePCA.fit(raw[:40])
    expected = sequence_examples(raw, adapter, [70])
    changed = raw.copy()
    changed[70:] += 100
    actual = sequence_examples(changed, adapter, [70])
    np.testing.assert_array_equal(expected["x"], actual["x"])
    assert not np.array_equal(expected["y"], actual["y"])
    assert expected["mask"].sum() == HISTORY


def test_identical_labels_generated_separately_from_whole_stream_and_drift_has_no_attacks():
    tasks = row_tasks(DOMAINS[0], 80)
    np.testing.assert_array_equal(tasks["identical_labels"][0], row_draw(80, DOMAINS[0], 8005))
    np.testing.assert_array_equal(
        tasks["identical_labels"][1], np.random.default_rng(8006).integers(0, 2, 80)
    )
    assert not tasks["normal_drift"][1].any()
    temporal, _ = temporal_tasks(DOMAINS[0], 80)
    expected, _ = temporal_draw(80 + HISTORY, DOMAINS[0], 9002)
    np.testing.assert_array_equal(temporal["identical_labels"][0], expected)
    assert not temporal["normal_drift"][1].any()


def test_normal_only_metrics_are_defined_without_fictitious_auc_or_tpr():
    result = finite_metrics(np.array([True, False]), np.array([2.0, 1.0]), np.zeros(2, int))
    assert result["attack_windows"] == 0
    assert result["tpr"] is None and result["auroc"] is None and result["pauc_005"] is None
    assert result["fpr"] == 0.5


def test_negative_control_formal_test_is_fixed_policy_only_not_forced_equal_rates():
    flags = np.array([True, True, False, False])
    y = np.array([1, 1, 0, 0])
    fixed = finite_metrics(flags, flags.astype(float), y, True, "fixed")
    assert fixed["tpr"] == 1.0 and fixed["fpr"] == 0.0
    assert 0 <= fixed["fixed_policy_fisher_exact_two_sided_p"] <= 1
    assert len(fixed["fixed_policy_null95_tpr_minus_fpr"]) == 2
    adaptive = finite_metrics(flags, flags.astype(float), y, True, "immediate")
    assert "fixed_policy_fisher_exact_two_sided_p" not in adaptive


def test_model_adapter_normal_fit_is_unaffected_by_target_draw_mutation():
    train = torch.from_numpy(row_draw(64, DOMAINS[0], 1001))
    target = row_draw(64, DOMAINS[0], 8001)
    expected = SourcePCA.fit(train).state_dict()
    target[:] = -1e6
    actual = SourcePCA.fit(train).state_dict()
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)


def test_h2_support_uses_training_rows_and_real_pipeline_sd_floor():
    train = torch.from_numpy(row_draw(64, DOMAINS[0], 1001))
    select = torch.from_numpy(row_draw(32, DOMAINS[0], 1002))
    adapter = SourcePCA.fit(train)
    model = make_model("mlp32", train.shape[1], 17)
    state = fit_row_state(model, "mlp32", adapter, train, select, 17, [])
    assert state["support_fit_role"] == "train" and state["support_fit_rows"] == 64
    assert state["geometry"]["counts"].sum() == 64
    with torch.no_grad():
        f = model.embed(model_input(adapter, train, "mlp32"))
    torch.testing.assert_close(state["mean"], f.mean(0), rtol=0, atol=0)
    torch.testing.assert_close(
        state["scale"], f.std(0, unbiased=False).clamp_min(1e-6), rtol=0, atol=0
    )


def test_h2_ensemble_selection_renormalization_and_disk_restore(tmp_path):
    train = torch.from_numpy(row_draw(64, DOMAINS[0], 1001))
    select = torch.from_numpy(row_draw(32, DOMAINS[0], 1002))
    adapter = SourcePCA.fit(train)
    seeds = [
        fit_row_state(
            make_model("mlp32", train.shape[1], seed), "mlp32", adapter, train, select, seed, []
        )
        for seed in (17, 29)
    ]
    selected = [row_scores(seed, select) for seed in seeds]
    state = {
        "seeds": seeds,
        "ensemble_units": {
            h: units(torch.from_numpy(np.mean([s[h] for s in selected], axis=0)))
            for h in ("scalar", "support")
        },
    }
    expected = row_ensemble_scores(state, select)
    for values in expected.values():
        np.testing.assert_allclose(np.quantile(values, [0.5, 0.95]), [1.0, 2.0], atol=1e-12)
    path = tmp_path / "ensemble.pt"
    torch.save(state, path)
    actual = row_ensemble_scores(torch.load(path, weights_only=True), select)
    for head in expected:
        np.testing.assert_array_equal(expected[head], actual[head])
