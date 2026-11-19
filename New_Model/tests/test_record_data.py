from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fedxmag_next.record_data import FEATURES, NUMERIC, _parse_file
from fedxmag_next.record_experiment import corruption_groups


def test_completed_connection_features_preserve_availability_and_time(tmp_path: Path) -> None:
    base = {name: "0" for name in NUMERIC}
    base.update(
        {
            "ts": "1000.25",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "proto": "tcp",
            "service": "http",
            "conn_state": "SF",
            "src_port": "50000",
            "dst_port": "80",
            "http_uri": "/index",
            "label": "0",
            "type": "normal",
        }
    )
    base["duration"] = "10.5"
    missing = {**base, "ts": "1100", "duration": "-", "src_bytes": "-"}
    path = tmp_path / "connections.csv"
    pd.DataFrame([base, missing]).to_csv(path, index=False)
    parsed = _parse_file((str(path), 0, 1))
    position = {name: index for index, name in enumerate(FEATURES)}
    assert parsed["observed_at"].tolist() == [1011, 1100]
    assert parsed["features"][1, position["available_src_bytes"]] == 0
    assert parsed["features"][0, position["available_src_bytes"]] == 1
    assert np.isfinite(parsed["features"]).all()
    assert "label" not in position and "ts" not in position and "src_ip" not in position
    groups = corruption_groups()
    numeric_pair = [position["log_src_bytes"], position["available_src_bytes"]]
    assert numeric_pair in groups


def test_explicit_quota_retains_self_observation_and_never_encodes_row_ids_or_labels(tmp_path):
    row = {name: "0" for name in NUMERIC}
    row.update(
        {
            "ts": "1000",
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
            "src_port": "50000",
            "dst_port": "80",
            "proto": "tcp",
            "service": "http",
            "conn_state": "SF",
            "http_uri": "/",
            "label": "0",
            "type": "normal",
        }
    )
    rows = [row, {**row, "ts": "1002", "dst_ip": "10.0.0.1", "label": "1"}, {**row, "ts": "1004"}]
    source = tmp_path / "rows.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    ordinary = _parse_file((str(source), 0, 1))
    assert ordinary["source_row_index"].tolist() == [0, 2]
    extra = tmp_path / "extra.npy"
    np.save(extra, np.array([1]))
    options = {"extra_rows": str(extra), "keep_self_connections": True}
    quota = _parse_file((str(source), 0, 1, options))
    assert quota["source_row_index"].tolist() == [0, 1, 2]
    assert quota["features"].shape[1] == len(FEATURES)
    np.testing.assert_array_equal(quota["features"][[0, 2]], ordinary["features"])
    for item in rows:
        item["label"] = str(1 - int(item["label"]))
    pd.DataFrame(rows).to_csv(source, index=False)
    flipped = _parse_file((str(source), 0, 1, options))
    np.testing.assert_array_equal(flipped["signature"], quota["signature"])
    np.testing.assert_array_equal(flipped["features"], quota["features"])
    np.testing.assert_array_equal(flipped["source_row_index"], quota["source_row_index"])
