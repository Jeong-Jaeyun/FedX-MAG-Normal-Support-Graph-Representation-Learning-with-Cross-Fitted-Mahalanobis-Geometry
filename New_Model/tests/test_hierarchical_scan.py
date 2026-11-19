import numpy as np
import pytest

from fedxmag_next.hierarchical_scan import group_statistics, grouping, scan_surprise, surprises


def test_nested_groups_preserve_identity_equality_and_are_permutation_invariant():
    keys = {
        "source": np.array([2, 2, 4, 4], dtype=np.uint64),
        "destination": np.array([9, 9, 9, 10], dtype=np.uint64),
    }
    assert sorted(map(len, grouping(keys, "source"))) == [2, 2]
    assert sorted(map(len, grouping(keys, "destination"))) == [1, 3]
    assert sorted(map(len, grouping(keys, "pair"))) == [1, 1, 2]
    values = {"edge": np.array([0.0, 1.0, 4.0, 6.0])}
    reference = {"edge": np.arange(5.0)}
    order = np.array([3, 0, 2, 1])
    for scope in ("source", "destination", "pair"):
        actual = group_statistics(values, keys, scope, reference)["edge"]
        changed = group_statistics(
            {k: v[order] for k, v in values.items()},
            {k: v[order] * 123 + 7 for k, v in keys.items()},
            scope,
            reference,
        )["edge"]
        np.testing.assert_allclose(np.sort(actual), np.sort(changed), atol=0, rtol=0)


def test_group_scan_keeps_local_sparse_evidence_and_handles_numeric_ties():
    # One suspicious endpoint retains its evidence despite many benign peers.
    scores = {"edge": np.r_[100.0, np.zeros(99)]}
    keys = {
        "source": np.r_[1, np.repeat(2, 99)].astype(np.uint64),
        "destination": np.ones(100, dtype=np.uint64),
    }
    reference = {"edge": np.arange(50.0)}
    grouped = group_statistics(scores, keys, "source", reference)["edge"]
    global_scan = scan_surprise(surprises(scores["edge"], reference["edge"]))
    assert grouped.max() > global_scan + 2
    baseline = surprises(np.array([2.0, 3.0]), np.arange(5.0))
    np.testing.assert_array_equal(baseline, surprises(np.array([2.0, 3.0]) + 1e-8, np.arange(5.0)))
    with pytest.raises(ValueError):
        grouping(keys, "unknown")
    with pytest.raises(ValueError):
        surprises(np.array([1.0]), np.array([]))
