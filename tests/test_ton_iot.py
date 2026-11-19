import csv
import math

import numpy as np
import pytest

from fedxmag.flow_context import NUMERIC_COLUMNS, WINDOW_NS
from fedxmag.preprocessing.events import read_xiiotid_events
from fedxmag.schema import RESOLUTION_AWARE_FEATURE_GROUPS
from fedxmag.ton_iot import capture_of, read_ton_records, session_key

START = 1_556_200_800  # 2019-04-25T14:00:00Z = 2019-04-26 00:00 AEST
HEADER = [
    "ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service", "duration",
    "src_bytes", "dst_bytes", "conn_state", "missed_bytes", "src_pkts", "src_ip_bytes",
    "dst_pkts", "dst_ip_bytes", "dns_query", "weird_notice", "label", "type",
]
RENAMED = {"ts": "Timestamp", "src_ip": "Scr_IP", "dst_ip": "Des_IP", "proto": "Protocol",
           "src_bytes": "Scr_bytes", "dst_bytes": "Des_bytes", "src_pkts": "Scr_pkts",
           "dst_pkts": "Des_pkts", "label": "class3", "type": "class2"}


def row(seconds, source, destination, *, proto="tcp", service="-", state="SF", port="80",
        src=("100", "4"), dst=("300", "3"), label="0", kind="normal", sport="40000"):
    values = {"ts": str(START + seconds), "src_ip": source, "src_port": sport,
              "dst_ip": destination, "dst_port": port, "proto": proto, "service": service,
              "duration": "2.0", "src_bytes": src[0], "dst_bytes": dst[0], "conn_state": state,
              "missed_bytes": "0", "src_pkts": src[1], "src_ip_bytes": "0", "dst_pkts": dst[1],
              "dst_ip_bytes": "0", "dns_query": "-", "weird_notice": "-", "label": label,
              "type": kind}
    return [values[name] for name in HEADER]


ROWS = [
    row(-5, "a", "b"),  # previous AEST day
    row(1, "a", "b"),
    row(1, "a", "b"),  # exact duplicate, removed
    row(1, "a", "b", sport="40001"),  # differs only in a column the graph ignores: kept
    row(2, "b", "a", proto="udp", service="dns", state="SF", port="53", src=("0", "1"),
        dst=("0", "0")),
    row(3, "a", "c", state="S0", src=("0", "1"), dst=("0", "0"), label="1", kind="scanning"),
    row(4, "a", "a"),  # self-relation, removed
    row(5, "-", "b"),  # missing endpoint, removed
    row(6, "c", "b", state="REJ", src=("-", "-"), dst=("-", "-"), label="1", kind="scanning"),
    row(61, "b", "c", state="OTH", service="ssl"),
]


@pytest.fixture
def tables(tmp_path):
    ton = tmp_path / "Network_dataset_1.csv"
    with ton.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(ROWS)
        writer.writerow(["not-a-time", *ROWS[1][1:]])  # bad timestamp, removed
    renamed = tmp_path / "renamed.csv"
    with renamed.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([RENAMED.get(name, name) for name in HEADER] + ["Date"])
        for values in [*ROWS, ["not-a-time", *ROWS[1][1:]]]:
            seconds = int(values[0]) if values[0].isdigit() else START
            writer.writerow([*values, capture_of(seconds * 1_000_000_000)])
    return ton, renamed


def test_reader_matches_frozen_x_iiotid_ingestion(tables) -> None:
    ton, renamed = tables
    records = read_ton_records([ton])
    assert records.report == {"rows_total": 11, "exact_duplicate": 1, "bad_timestamp": 1,
                              "missing_endpoint": 1, "self_relation": 1, "kept": 7}
    reference = read_xiiotid_events(renamed)
    assert reference.report.rows_skipped_exact_duplicate == 1
    expected = sorted(
        (e.timestamp_ns, e.src_key, e.dst_key, e.directional_mean_packet_size, e.protocol,
         e.is_attack, e.attack_category, e.timing_observed, e.session_key.split(":")[-1])
        for e in reference.events
    )
    actual = sorted(
        (e.timestamp_ns, e.src_key, e.dst_key, e.directional_mean_packet_size, e.protocol,
         e.is_attack, e.attack_category, e.timing_observed, e.session_key.split(":")[-1])
        for e in records.events(range(len(records)))
    )
    assert actual == expected
    assert {e.session_key for e in records.events(range(len(records)))} == {
        session_key("2019-04-25"), session_key("2019-04-26")
    }


def test_flags_follow_zeek_states_and_views_split_by_label(tables) -> None:
    records = read_ton_records([tables[0]])
    position = {name: column for column, name in enumerate(NUMERIC_COLUMNS)}
    flags = records.flow_records(range(len(records))).values
    # Kept rows: SF tcp, SF tcp, SF tcp, udp, S0 tcp, REJ tcp, OTH tcp.
    syn, fin = position["is_syn_only"], position["FIN or RST"]
    ack, syn_ack = position["is_pure_ack"], position["Is_SYN_ACK"]
    payload = position["is_with_payload"]
    np.testing.assert_array_equal(flags[:, syn], [1, 1, 1, 0, 1, 1, 0])
    np.testing.assert_array_equal(flags[:, fin], [1, 1, 1, 0, 0, 1, 0])
    np.testing.assert_array_equal(flags[:, ack], [1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(flags[:, syn_ack], [1, 1, 1, 0, 0, 0, 0])
    assert flags[3, payload] == 0.0 and math.isnan(flags[5, payload])
    services = records.flow_records(range(len(records))).services
    assert services == ["other", "other", "other", "dns", "other", "other", "ssl"]

    windows = records.windows()
    first = (START * 1_000_000_000) // WINDOW_NS
    assert sorted(windows) == [first - 1, first, first + 1]
    rows = windows[first]
    settings = {"timing_profile": "distinct_second_collision_v3",
                "timestamp_resolution_seconds": 1}
    graph, flows, keys = records.view(rows, RESOLUTION_AWARE_FEATURE_GROUPS, **settings)
    assert graph.label == 1 and graph.metadata["session_key"] == session_key("2019-04-26")
    assert graph.metadata["endpoint_keys"] == ["a", "b", "c"]
    assert sorted(keys) == [("a", "b"), ("a", "c"), ("b", "a"), ("c", "b")]
    background = rows[~records.attack[rows]]
    graph, flows, keys = records.view(background, RESOLUTION_AWARE_FEATURE_GROUPS, **settings)
    assert graph.label == 0 and sorted(keys) == [("a", "b"), ("b", "a")]
    assert flows.shape[0] == 2 and graph.metadata["event_count"] == 5
