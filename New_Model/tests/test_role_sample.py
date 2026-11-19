import numpy as np

from fedxmag_next.native_context import EVENT_DTYPE
from fedxmag_next.role_sample import missing_role_rows


def test_role_sampling_covers_missing_directions_without_labels_or_self_queries():
    events = np.zeros(6, dtype=EVENT_DTYPE)
    events["source"] = [1, 1, 2, 2, 3, 4]
    events["destination"] = [8, 9, 8, 9, 3, 9]
    events["query"] = [0, -1, -1, -1, -1, -1]
    rows = np.array([10, 11, 12, 13, 14, 15])
    chosen = missing_role_rows(events, rows)
    assert 14 not in chosen
    query = np.isin(rows, chosen) | (events["query"] >= 0)
    eligible = events["source"] != events["destination"]
    for field in ("source", "destination"):
        assert set(events[field][eligible]) == set(events[field][query])
    perm = np.array([4, 1, 5, 0, 3, 2])
    assert np.array_equal(chosen, missing_role_rows(events[perm], rows[perm]))
