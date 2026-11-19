import numpy as np
import pandas as pd

from fedxmag_next.window_bridge import (
    aggregate_connections,
    census,
    complete_role_windows,
    replay_windows,
)


def test_whole_window_truth_includes_all_rows_and_unions_across_files(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame(
        {"ts": [100, 110], "duration": [0, 20], "label": [0, 1], "type": ["normal", "one"]}
    ).to_csv(first, index=False)
    pd.DataFrame(
        {
            "Timestamp": [130.5, 179.5],
            "Duration": [0, 1],
            "class3": ["Attack", "Normal"],
            "class2": ["two", "Normal"],
        }
    ).to_csv(second, index=False)
    truth = census([first, second])
    assert truth[1] == {"label": 0, "records": 1, "types": []}
    assert truth[2] == {"label": 1, "records": 2, "types": ["one", "two"]}
    assert truth[3]["label"] == 0
    assert np.isclose(sum(item["records"] for item in truth.values()), 4)


def test_source_calibration_drops_both_partial_boundary_windows():
    times = np.array([0, 1, 2, 60, 61, 120, 121])
    truth = {minute: {"label": 0} for minute in range(3)}
    assert complete_role_windows(times, np.array([1, 2, 3, 4, 5]), truth) == [1]
    truth[1]["label"] = 1
    assert complete_role_windows(times, np.array([1, 2, 3, 4, 5]), truth) == []


def test_continuous_window_feedback_survives_date_boundary_and_query_is_scored_first():
    scores = np.r_[np.arange(20), 100.0]
    labels = np.r_[np.zeros(20, dtype=int), 1]
    captures = np.r_[np.ones(20, dtype=int), 2]
    observed = np.ones(21, dtype=bool)
    flags, valid = replay_windows(scores, observed, labels, captures, [], policy="continuous")
    assert not valid[:19].any()
    assert valid[19] and flags[19]
    assert valid[20] and flags[20]
    _, reset_valid = replay_windows(scores, observed, labels, captures, [])
    assert not reset_valid[20]


def test_rank_aggregation_is_monotone_invariant_and_conservative_with_ties():
    reference = np.arange(30, dtype=float)
    query = np.array([2.0, 20.0, 40.0])
    for method in ("rank_scan", "rank_mean"):
        assert np.isclose(
            aggregate_connections(query, reference, method),
            aggregate_connections(np.exp(query), np.exp(reference), method),
        )
        assert aggregate_connections(np.ones(4), np.ones(30), method) == 0


def test_tail_extension_separates_outliers_above_reference_and_preserves_affine_units():
    reference = np.arange(100, dtype=float)
    low, high = np.array([200.0]), np.array([1000.0])
    assert aggregate_connections(low, reference, "rank_scan") == aggregate_connections(
        high, reference, "rank_scan"
    )
    assert aggregate_connections(high, reference, "tail_scan") > aggregate_connections(
        low, reference, "tail_scan"
    )
    assert np.isclose(
        aggregate_connections(low, reference, "tail_scan"),
        aggregate_connections(low * 2 + 50, reference * 2 + 50, "tail_scan"),
    )
    assert aggregate_connections(np.ones(5), np.ones(100), "tail_scan") == 0
