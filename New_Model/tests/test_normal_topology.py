import numpy as np
import pytest

from fedxmag_next.normal_topology import NormalTopology


def test_topology_predicts_before_normal_admission_and_bounds_categorical_memory():
    familiar = (1, 2, 80, 0, 1)
    novel = (1, 3, 443, 0, 1)
    model = NormalTopology([familiar] * 4, 10, maximum=4)
    before = model.score([novel])
    np.testing.assert_array_equal(model.query(20, [novel], [15]), before)
    with pytest.raises(ValueError, match="feedback"):
        model.state_dict()
    model.feedback(verified_normal=False)
    np.testing.assert_array_equal(model.score([novel]), before)
    model.query(30, [novel] * 4, [21, 22, 23, 24])
    model.feedback(verified_normal=True)
    assert len(model.rows) == 4 and not model.global_counts[0][2]
    assert (model.score([novel]) < before).all()
    restored = NormalTopology.from_state_dict(model.state_dict())
    np.testing.assert_array_equal(restored.score([familiar, novel]), model.score([familiar, novel]))


def test_topology_uses_key_equality_and_global_backoff_for_unseen_source():
    rows = [(1, 2, 80, 0, 1), (1, 3, 443, 0, 1), (4, 2, 80, 0, 1)]
    renamed = [(a + 100, b + 200, c, d, e) for a, b, c, d, e in rows]
    first = NormalTopology(rows, 10)
    second = NormalTopology(renamed, 10)
    np.testing.assert_array_equal(first.score(rows), second.score(renamed))
    assert np.isfinite(first.score([(999, 888, -1, 3, 0)])).all()
    with pytest.raises(ValueError, match="current completed interval"):
        first.query(20, rows, [21, 21, 21])
