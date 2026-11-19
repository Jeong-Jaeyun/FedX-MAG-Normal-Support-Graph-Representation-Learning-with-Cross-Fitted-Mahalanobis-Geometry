import numpy as np

from fedxmag_next.commission_roles import early_normal_plans, latest_ready_stage


def test_commissioning_waits_for_whole_normal_feedback_and_raw_start_purge():
    minutes = np.arange(100, 125)
    observed = np.repeat(minutes * 60 + 40, 2)
    started = observed.copy()
    labels = np.zeros(len(minutes), dtype=int)
    labels[1] = 1
    # This later observed window has one long flow predating the train cut.
    started[np.repeat(minutes == 104, 2)] = 6000
    data = {"observed_at": observed, "timestamp": started, "utc_offset_seconds": np.array(0)}
    plan = early_normal_plans(data, minutes, labels, count=2, gap=60, delay=120)[0]
    assert plan["train_minutes"] == [100, 102]
    assert plan["selection_minutes"] == [105, 106]
    assert plan["ready_at"] == 107 * 60 + 120
    assert latest_ready_stage(plan["ready_at"], [plan["ready_at"]]) == 0
    assert latest_ready_stage(plan["ready_at"] + 1, [plan["ready_at"]]) == 1
    assert not np.isin(np.flatnonzero(np.repeat(labels == 1, 2)), plan["train_rows"]).any()


def test_future_normal_feedback_cannot_change_earlier_commissioning_availability():
    minutes = np.arange(100, 125)
    data = {
        "observed_at": minutes * 60 + 40,
        "timestamp": minutes * 60 + 40,
        "utc_offset_seconds": 0,
    }
    original = early_normal_plans(data, minutes, np.zeros(25), count=2, gap=60, delay=120)
    labels = np.zeros(25)
    labels[10:] = 1
    modified = early_normal_plans(data, minutes, labels, count=2, gap=60, delay=120)
    assert original[0]["ready_at"] == modified[0]["ready_at"]
    assert latest_ready_stage(0, [original[0]["ready_at"]]) == 0
