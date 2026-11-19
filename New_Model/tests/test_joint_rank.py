import numpy as np
import pytest

from fedxmag_next.joint_rank import joint_rank_pvalue


@pytest.mark.parametrize("pooling", ["any", "mean", "median", "all"])
def test_joint_rank_is_permutation_symmetric_and_conservative_for_any_fixed_batch(pooling):
    random = np.random.default_rng(62)
    for values in (random.normal(size=(23, 2)), random.integers(0, 5, (23, 2))):
        pvalues = np.asarray(
            [
                joint_rank_pvalue(np.delete(values, i, axis=0), values[i], pooling=pooling)
                for i in range(len(values))
            ]
        )
        for alpha in np.arange(1, 24) / 23:
            assert np.mean(pvalues <= alpha) <= alpha + 1e-12
        ref, query = values[:-1], values[-1]
        expected = joint_rank_pvalue(ref, query, pooling=pooling)
        assert expected == joint_rank_pvalue(ref[::-1], query, pooling=pooling)
        assert expected == joint_rank_pvalue(np.exp(ref), np.exp(query), pooling=pooling)
        assert expected == joint_rank_pvalue(ref[:, ::-1], query[::-1], pooling=pooling)


def test_single_head_matches_conservative_upper_tail_count_including_query():
    ref = np.array([[1.0], [2.0], [2.0], [5.0]])
    for query in (0.0, 2.0, 3.0, 6.0):
        expected = (1 + np.sum(ref[:, 0] >= query)) / (len(ref) + 1)
        assert joint_rank_pvalue(ref, np.array([query])) == expected
