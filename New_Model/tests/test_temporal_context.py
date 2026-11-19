import numpy as np
import pandas as pd

from fedxmag_next.record_data import FEATURES, XIIOTID_COLUMNS, _parse_file
from fedxmag_next.record_experiment import corruption_groups
from fedxmag_next.temporal_context import CONTEXT_FEATURES, describe_events, extract_day


def test_context_excludes_current_second_and_future_and_expires_old_events():
    events = [
        (10, 1, 2, 80, 0, 100, 2),
        (10, 1, 3, 22, 1, 50, 1),
        (11, 1, 2, 80, 0, 10, 1),
        (71, 1, 2, 80, 0, 10, 1),
        (72, 1, 2, 80, 0, 10, 1),
    ]
    got = describe_events(events, np.arange(5))
    assert np.all(got[:2] == 0)
    assert np.isclose(got[2, 0], np.log1p(2))
    assert np.isclose(got[2, 1], np.log1p(2))
    assert got[2, 3] == 0.5
    assert np.isclose(got[3, 0], np.log1p(1))
    assert np.isclose(got[4, 0], np.log1p(1))
    np.testing.assert_array_equal(got[:3], describe_events(events[:3], np.arange(3)))


def test_identity_renaming_does_not_change_features_and_context_corruption_is_joint():
    events = [(10, 1, 2, 80, 0, 10, 1), (11, 2, 1, 80, 1, 20, 2)]
    renamed = [(t, a + 100, b + 100, p, f, n, k) for t, a, b, p, f, n, k in events]
    np.testing.assert_array_equal(
        describe_events(events, np.arange(2)), describe_events(renamed, np.arange(2))
    )
    groups = corruption_groups((*FEATURES, "payload_http_observed", *CONTEXT_FEATURES))
    assert groups[-1] == list(range(len(FEATURES) + 1, len(FEATURES) + 1 + len(CONTEXT_FEATURES)))
    assert len(FEATURES) in groups[-2]


def test_xiiotid_context_join_matches_existing_observation_signature(tmp_path):
    rows = []
    for timestamp in (1000, 1002, "?"):
        row = {
            "ts": str(timestamp),
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": "40000",
            "dst_port": "80",
            "proto": "tcp",
            "service": "http",
            "conn_state": "SF",
            "duration": "1",
            "src_bytes": "10",
            "dst_bytes": "20",
            "src_pkts": "1",
            "dst_pkts": "2",
            "src_ip_bytes": "50",
            "dst_ip_bytes": "100",
            "label": "Normal",
            "type": "Normal",
        }
        inverse = {value: key for key, value in XIIOTID_COLUMNS.items()}
        rows.append({inverse[key]: value for key, value in row.items()})
    source, prepared, output = tmp_path / "x.csv", tmp_path / "records.npz", tmp_path / "ctx.npz"
    pd.DataFrame(rows).to_csv(source, index=False)
    parsed = _parse_file((str(source), 0, 1))
    np.savez(prepared, signature=parsed["signature"], dataset="xiiotid")
    report = extract_day((str(source), str(prepared), str(output)))
    assert report["unique_queries"] == 2
    assert report["skipped_invalid_time"] == 1
    with np.load(output) as data:
        got = {int(key): row for key, row in zip(data["signature"], data["context"], strict=True)}
        assert got[int(parsed["signature"][0])][0] == 0
        assert np.isclose(got[int(parsed["signature"][1])][0], np.log1p(1))
