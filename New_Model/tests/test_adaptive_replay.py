import numpy as np

from fedxmag_next.adaptive_replay import replay_adaptive


def test_new_model_waits_until_ready_and_does_not_calibrate_on_fitting_feedback():
    minutes = np.arange(100, 106)
    labels = np.array([0, 0, 1, 0, 1, 1])
    observed = np.ones(6, dtype=bool)
    base = np.full(6, 100.0)
    stage = {
        "ready_at": 102 * 60,
        "calibration": [0.0] * 39,
        "energy": np.full(6, 2.0),
        "admit": np.array([False, False, False, True, True, True]),
    }
    result = replay_adaptive(minutes, labels, observed, base, [0.0] * 39, [stage])
    assert result["model_assignment"].tolist() == [0, 0, 1, 1, 1, 1]
    assert result["threshold"][2] == 0
    assert result["flags"][2]
    assert result["threshold"][4] == 0
    changed = {**stage, "energy": np.array([999.0, 999.0, 2.0, 2.0, 2.0, 2.0])}
    replay = replay_adaptive(minutes, labels, observed, base, [0.0] * 39, [changed])
    assert np.array_equal(result["flags"], replay["flags"])


def test_future_normal_feedback_cannot_change_earlier_adaptive_predictions():
    minutes = np.arange(100, 110)
    observed = np.ones(10, dtype=bool)
    energy = np.arange(10, dtype=float)
    stage = {
        "ready_at": 103 * 60,
        "calibration": [0.0] * 39,
        "energy": energy + 1,
        "admit": observed,
    }
    labels = np.ones(10, dtype=int)
    first = replay_adaptive(minutes, labels, observed, energy, [0.0] * 39, [stage])
    labels[7:] = 0
    second = replay_adaptive(minutes, labels, observed, energy, [0.0] * 39, [stage])
    for key in ("flags", "valid", "energy", "threshold", "model_assignment"):
        np.testing.assert_array_equal(first[key][:8], second[key][:8])
