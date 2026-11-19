from __future__ import annotations

import numpy as np

from fedxmag_next.record_calibration import replay


def test_same_second_query_cannot_use_another_querys_normal_feedback() -> None:
    values = np.asarray([100.0, 50.0, 50.0])
    labels = np.asarray([0, 1, 1])
    captures = np.asarray([1, 1, 1])
    timestamp = np.asarray([10, 10, 11])
    cal = np.zeros(19)
    cal_capture = np.ones(19, dtype=int)
    cal_time = np.arange(-19, 0)
    flags, valid = replay(values, labels, captures, timestamp, cal, cal_capture, cal_time, 20)
    assert valid.all()
    assert flags.tolist() == [True, True, False]
