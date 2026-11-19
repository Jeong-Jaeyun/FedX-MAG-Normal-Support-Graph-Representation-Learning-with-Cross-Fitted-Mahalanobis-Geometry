import numpy as np
import pytest

from fedxmag_next.merge_cic_observations import merge_arrays
from fedxmag_next.record_data import FEATURES
from fedxmag_next.temporal_context import CONTEXT_FEATURES


def test_coverage_expansion_preserves_old_observations_and_does_not_invent_http():
    names = list(FEATURES) + list(CONTEXT_FEATURES)
    expanded = {
        "signature": np.array([1, 2, 3], dtype=np.uint64),
        "features": np.ones((3, len(names)), dtype=np.float32),
        "feature_names": np.array(names),
    }
    for field in ("timestamp", "observed_at", "label", "type", "capture"):
        expanded[field] = np.arange(3)
    legacy = {
        k: expanded[k][[0, 2]].copy()
        for k in ("signature", "timestamp", "observed_at", "label", "type", "capture")
    }
    legacy["feature_names"] = np.array(names + ["payload_http_observed"])
    legacy["features"] = np.c_[expanded["features"][[0, 2]], np.ones(2)].astype(np.float32)
    legacy["http_features"] = np.ones((2, 1), dtype=np.float32)
    legacy["http_feature_names"] = np.array(["http_observed"])
    result, positions = merge_arrays(expanded, legacy)
    assert np.array_equal(result["features"][positions], legacy["features"])
    assert result["features"][1, -1] == 0 and result["http_features"][1, 0] == 0
    bad = {**expanded, "features": expanded["features"].copy()}
    bad["features"][0, 0] = 2
    with pytest.raises(ValueError, match="old non-HTTP"):
        merge_arrays(bad, legacy)
