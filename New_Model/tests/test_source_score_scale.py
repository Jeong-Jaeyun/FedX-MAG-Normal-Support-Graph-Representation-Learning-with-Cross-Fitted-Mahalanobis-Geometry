import numpy as np

from fedxmag_next.source_score_scale import SourceScoreScale


def test_global_score_scale_is_exact_identity_for_control_replay():
    values = np.array([0.1, 0.3, 0.7, 1.001, 2.9], dtype=np.float64)
    keys = [(1, 80, 0, 1)] * len(values)
    model = SourceScoreScale.fit(values, keys, "global", minimum=2, global_q95=5.0)
    np.testing.assert_array_equal(model.transform(values, keys), values)


def test_local_score_units_use_equality_and_service_then_global_backoff():
    values = np.array([1.0] * 4 + [2.0] * 4 + [4.0] * 4)
    keys = [(1, 80, 0, 1)] * 4 + [(2, 80, 0, 1)] * 4 + [(3, 443, 0, 1)] * 4
    model = SourceScoreScale.fit(values, keys, "source_service", minimum=4, shrinkage=4)
    renamed = [(key[0] + 1000, *key[1:]) for key in keys]
    other = SourceScoreScale.fit(values, renamed, "source_service", minimum=4, shrinkage=4)
    np.testing.assert_array_equal(model.transform(values, keys), other.transform(values, renamed))
    transformed = model.transform(
        np.array([4.0, 4.0, 4.0, 4.0]),
        [(1, 80, 0, 1), (2, 80, 0, 1), (999, 80, 0, 1), (999, 999, 0, 1)],
    )
    assert transformed[0] > transformed[1] > transformed[2] > transformed[3] == 4
    restored = SourceScoreScale(model.state_dict())
    np.testing.assert_array_equal(restored.transform(values, keys), model.transform(values, keys))
