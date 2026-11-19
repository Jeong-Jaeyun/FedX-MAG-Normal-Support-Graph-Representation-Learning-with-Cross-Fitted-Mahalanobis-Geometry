import shutil

import numpy as np
import pytest

from fedxmag_next.native_context import EVENT_DTYPE, native_features
from fedxmag_next.temporal_context import describe_events


@pytest.mark.skipif(shutil.which("g++") is None, reason="local native compiler unavailable")
def test_native_history_matches_python_reference_with_expiration_duplicates_and_same_seconds():
    random = np.random.default_rng(51)
    events = []
    for timestamp in np.sort(random.integers(0, 2000, 3000)):
        events.append(
            (
                int(timestamp),
                int(random.integers(0, 7)),
                int(random.integers(0, 7)),
                int(random.choice([22, 80, 443])),
                int(random.integers(0, 2)),
                float(random.integers(0, 10000)),
                float(random.integers(1, 100)),
            )
        )
    queries = np.arange(0, len(events), 3)
    expected = describe_events(events, queries)
    packed = np.empty(len(events), dtype=EVENT_DTYPE)
    query_map = {int(row): index for index, row in enumerate(queries)}
    for i, event in enumerate(events):
        packed[i] = (*event, query_map.get(i, -1))
    result, seen = native_features(packed, len(queries))
    assert seen.all()
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="sort/bounds"):
        native_features(packed[::-1], len(queries))
