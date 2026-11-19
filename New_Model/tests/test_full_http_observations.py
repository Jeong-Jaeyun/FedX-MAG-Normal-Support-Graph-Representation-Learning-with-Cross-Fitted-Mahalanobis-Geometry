import numpy as np
import pytest

from fedxmag_next.full_http_observations import URI_COLUMNS, fill_arrays
from fedxmag_next.pcap_http import HTTP_FEATURES, describe_http


def _observations():
    names = ["duration", *URI_COLUMNS, *[f"payload_{name}" for name in HTTP_FEATURES]]
    values = np.zeros((3, len(names)), dtype=np.float32)
    values[:, 0] = [1, 2, 3]
    summaries = np.zeros((3, len(HTTP_FEATURES)), dtype=np.float32)
    summaries[1] = describe_http(b"GET /?q=%3C HTTP/1.1\r\nHost: test\r\n\r\n", b"")
    return {
        "signature": np.array([11, 22, 33], dtype=np.uint64),
        "features": values,
        "feature_names": np.array(names),
        "http_summary_inspected": np.array([True, False, False]),
        "label": np.array([0, 1, 0]),
        "observed_at": np.array([50, 60, 70]),
    }, summaries


def test_full_http_fills_actual_new_observations_and_preserves_query_identity():
    data, summaries = _observations()
    before = {key: value.copy() for key, value in data.items()}
    result = fill_arrays(data, summaries)
    for key in ("signature", "label", "observed_at", "feature_names"):
        np.testing.assert_array_equal(result[key], data[key])
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])
    np.testing.assert_array_equal(result["features"][:, 0], [1, 2, 3])
    np.testing.assert_array_equal(result["features"][0], data["features"][0])
    assert result["http_summary_inspected"].all()
    positions = {name: i for i, name in enumerate(result["feature_names"])}
    assert result["features"][1, positions["available_http_uri"]] == 1
    assert result["features"][1, positions["payload_http_observed"]] == 1
    assert result["features"][1, positions["log_http_uri_length"]] == summaries[1, 2]
    assert not result["features"][2, 1:].any()  # inspected, with no observed HTTP


def test_full_http_rejects_changed_previously_inspected_vector():
    data, summaries = _observations()
    summaries[0] = summaries[1]
    with pytest.raises(ValueError, match="previously inspected"):
        fill_arrays(data, summaries)
