import numpy as np
import pytest

from fedxmag_next.causal_risk import CausalPeakMemory, scan_contributors
from fedxmag_next.window_bridge import aggregate_connections


def test_causal_risk_needs_prior_evidence_and_normal_feedback_resets_only_current_keys():
    memory = CausalPeakMemory(0)
    assert memory.predict(60, ["a"], [10], [50]).tolist() == [10]
    with pytest.raises(ValueError, match="Finish feedback"):
        memory.predict(61, ["a"], [0], [60])
    memory.feedback(verified_normal=False, contributors=[0])
    assert memory.predict(120, ["b"], [0], [110])[0] == 0
    memory.feedback(verified_normal=True, contributors=[])
    assert memory.predict(180, ["a"], [0], [170])[0] > 0
    memory.feedback(verified_normal=True, contributors=[])
    assert memory.predict(240, ["a"], [0], [230]).tolist() == [0]
    memory.feedback(verified_normal=False, contributors=[])
    assert not memory.states


def test_augmented_scores_never_refresh_risk_without_new_base_evidence():
    memory = CausalPeakMemory(0, horizon=300)
    memory.predict(60, ["a"], [10], [50])
    memory.feedback(verified_normal=False, contributors=[0])
    assert memory.predict(180, ["a"], [0], [170])[0] > 0
    memory.feedback(verified_normal=False, contributors=[])
    assert memory.predict(360, ["a"], [0], [350])[0] == 0
    memory.feedback(verified_normal=False, contributors=[])


def test_scan_contributors_are_the_actual_maximizing_prefix_and_budget_is_bounded():
    values = np.array([10.0, 1.0, 8.0, 2.0])
    reference = np.linspace(0, 9, 40)
    members = scan_contributors(values, reference)
    k = len(members)
    p = (1 + np.sum(reference >= np.sort(values)[-k])) / (len(reference) + 1)
    expected = np.log(k / len(values)) - np.log(p)
    assert expected == pytest.approx(aggregate_connections(values, reference, "rank_scan"))
    memory = CausalPeakMemory(0, maximum_keys=2)
    memory.predict(60, ["a", "b", "c"], [10, 20, 30], [50, 51, 52])
    memory.feedback(verified_normal=False, contributors=[0, 1, 2])
    assert len(memory.states) == memory.maximum_seen == 2
    top = scan_contributors(values, reference, "top3")
    assert values[top].mean() == aggregate_connections(values, reference, "top3")


def test_lower_recent_evidence_survives_after_an_older_larger_peak_expires():
    memory = CausalPeakMemory(0)
    memory.predict(60, ["a"], [10], [50])
    memory.feedback(verified_normal=False, contributors=[0])
    memory.predict(180, ["a"], [6], [170])
    memory.feedback(verified_normal=False, contributors=[0])
    assert memory.predict(360, ["a"], [0], [350])[0] == pytest.approx(6 * np.exp(-190 / 300))
    memory.feedback(verified_normal=False, contributors=[])


def test_checkpoint_resumes_same_predictions_and_does_not_share_mutable_state(tmp_path):
    import torch

    memory = CausalPeakMemory(0)
    memory.predict(60, [(2**64 - 1, 7)], [10], [50])
    with pytest.raises(ValueError, match="committed"):
        memory.state_dict()
    memory.feedback(verified_normal=False, contributors=[0])
    path = tmp_path / "risk.pt"
    torch.save(memory.state_dict(), path)
    restored = CausalPeakMemory.from_state_dict(torch.load(path, weights_only=True))
    a = memory.predict(120, [(2**64 - 1, 7)], [0], [110])
    b = restored.predict(120, [(2**64 - 1, 7)], [0], [110])
    np.testing.assert_array_equal(a, b)
    restored.feedback(verified_normal=True, contributors=[])
    memory.feedback(verified_normal=False, contributors=[])
    assert memory.states and not restored.states


def test_actual_risk_replay_never_uses_current_or_future_feedback_before_prediction(tmp_path):
    from fedxmag_next.risk_experiment import replay_risk

    values = np.array([1.0, 20.0, 0.0, 0.0])
    keys = {"source": np.ones(4, dtype=np.uint64), "destination": np.full(4, 2, dtype=np.uint64)}
    groups = {i: np.array([i]) for i in range(4)}
    arguments = (
        values,
        np.arange(10.0),
        keys,
        np.array([10, 70, 130, 190]),
        groups,
        np.array([1, 2, 3]),
        np.array([0]),
    )
    first, diagnostic = replay_risk(
        *arguments,
        np.array([1, 1, 1]),
        np.array([True, False, False]),
        "source",
        tmp_path / "first.pt",
    )
    changed, _ = replay_risk(
        *arguments,
        np.array([1, 0, 1]),
        np.array([True, False, False]),
        "source",
        tmp_path / "changed.pt",
    )
    assert first[1] == changed[1] and first[2] == changed[2]
    assert first[3] > changed[3]
    assert diagnostic["checkpoint_resumed_exact"]
    ignored, _ = replay_risk(
        *arguments,
        np.array([1, 0, 1]),
        np.array([True, False, False]),
        "source",
        tmp_path / "no_clear.pt",
        clear_on_normal=False,
    )
    assert ignored == first
    guarded, _ = replay_risk(
        *arguments,
        np.array([1, 0, 1]),
        np.array([True, False, False]),
        "source",
        tmp_path / "guarded.pt",
        clear_on_normal=False,
        current_gate_quantile=0.95,
    )
    assert guarded[1] == first[1]
    assert guarded[2] == guarded[3] == 0
