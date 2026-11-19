import numpy as np
import pandas as pd
import pytest

from fedxmag_next.native_context import EVENT_DTYPE
from fedxmag_next.protocol_context import (
    FLAG_COLUMNS,
    TAGS,
    RollingProtocols,
    compile_library,
    event_flags,
    native_protocol_features,
)


def test_protocol_taxonomy_uses_observations_and_ignores_labels():
    empty = {name: "-" for name in FLAG_COLUMNS}
    frame = pd.DataFrame(
        [
            empty | {"proto": "icmp", "src_port": "5"},
            empty | {"proto": "udp", "service": "dns", "dns_query": "x.example", "dns_rcode": "3"},
            empty | {"proto": "tcp", "service": "ssl", "ssl_established": "F"},
            empty
            | {"proto": "tcp", "service": "http", "http_method": "GET", "http_status_code": "404"},
        ]
    )
    flags = event_flags(frame)
    assert flags.shape == (4, 12)
    for row, tag in enumerate(("icmp_redirect", "dns_error", "tls_unestablished", "http_error")):
        assert flags[row, TAGS.index(tag)] == 1
    assert flags.sum() == 4
    frame["label"], frame["type"] = 1, "attack"
    assert np.array_equal(event_flags(frame), flags)


def test_native_protocol_history_matches_reference_with_same_second_and_expiry(tmp_path):
    library = tmp_path / "protocol.dll"
    compile_library(library)
    events = np.zeros(7, dtype=EVENT_DTYPE)
    events["time"] = [100, 100, 160, 161, 240, 400, 401]
    events["source"] = [1, 1, 1, 2, 1, 1, 1]
    events["query"] = np.arange(7)
    flags = np.zeros((7, 12), dtype=np.uint8)
    flags[np.arange(7), [0, 1, 2, 3, 4, 5, 6]] = 1
    actual = native_protocol_features(events, flags, 7, library)
    expected = np.zeros_like(actual)
    histories = [RollingProtocols(h, 12) for h in (60, 300)]
    start = 0
    while start < len(events):
        stop = start + 1
        while stop < len(events) and events["time"][stop] == events["time"][start]:
            stop += 1
        for history in histories:
            history.advance(int(events["time"][start]))
        for i in range(start, stop):
            expected[i] = np.concatenate([h.query(int(events["source"][i])) for h in histories])
        for i in range(start, stop):
            for history in histories:
                history.add(int(events["time"][i]), int(events["source"][i]), flags[i])
        start = stop
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert not actual[:2].any()
    assert actual[2, 1] == 0.5
    bad = events.copy()
    bad["time"][3] = 90
    with pytest.raises(ValueError, match="chronological"):
        native_protocol_features(bad, flags, 7, library)
