import csv
import math

import pytest
import torch

from fedxmag.flow_context import (
    FLOW_DIM,
    NUMERIC_COLUMNS,
    WINDOW_NS,
    flow_table,
    flows_for,
    read_flow_records,
)
from fedxmag.synthetic import make_synthetic_graphs

START = 1_577_836_800  # 2020-01-01T00:00:00Z, exactly on a window boundary
COLUMNS = ["Date", "Timestamp", "Scr_IP", "Des_IP", "Protocol", "Service", *NUMERIC_COLUMNS,
           "class3"]


def row(seconds, source, destination, protocol="tcp", service="http", port="80", label="Normal"):
    values = {"is_syn_only": "false", "FIN or RST": "true", "is_with_payload": "true",
              "is_pure_ack": "-", "Is_SYN_ACK": "false", "Duration": "2.0", "Des_port": port,
              "Scr_bytes": "100", "Des_bytes": "300"}
    return [
        "2020-01-01", str(START + seconds), source, destination, protocol, service,
        *(values[name] for name in NUMERIC_COLUMNS), label,
    ]


@pytest.fixture
def capture(tmp_path):
    path = tmp_path / "capture.csv"
    rows = [
        row(1, "a", "b"),
        row(1, "a", "b"),  # exact duplicate, removed
        row(2, "a", "b", protocol="udp", service="dns", port="5353"),
        row(3, "b", "a"),
        row(4, "a", "a"),  # self-relation, removed
        row(61, "a", "c", label="Attack"),
    ]
    rows.append([*rows[0][:1], "not-a-time", *rows[0][2:]])  # bad timestamp, removed
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return path


def test_flow_context_follows_ingestion_rules_and_edge_summaries(capture) -> None:
    records = read_flow_records(capture)
    assert len(records) == 4
    session = "file-0000:capture.csv:2020-01-01"
    assert set(records.sessions) == {session}
    first = (session, START * 1_000_000_000 // WINDOW_NS)
    table = flow_table(records)
    assert set(table) == {first, (first[0], first[1] + 1)}
    flows, keys = table[first]
    assert keys == [("a", "b"), ("b", "a")]
    assert flows.shape == (2, FLOW_DIM) and flows.dtype == torch.float64
    expected = [
        math.log1p(2), 0.5, 0.5, 0.0, 0.5, 0.5, 0.0,
        0.0, 1.0, 1.0, 0.0, 0.0,  # the unavailable pure-ACK flag averages to 0
        math.log1p(2.0), math.log1p(2), 0.5, math.log1p(100.0), math.log1p(300.0),
    ]
    torch.testing.assert_close(flows[0], torch.tensor(expected, dtype=torch.float64))
    assert set(flow_table(records, {first})) == {first}


def test_flows_align_with_graph_windows_and_refuse_missing_context(capture) -> None:
    table = flow_table(read_flow_records(capture))
    graphs = make_synthetic_graphs(num_graphs=2, seed=3)
    for offset, graph in enumerate(graphs):
        graph.metadata.update(
            session_key="file-0000:capture.csv:2020-01-01",
            window_start_ns=(START + 60 * offset) * 1_000_000_000,
        )
    flows, keys = flows_for(graphs[::-1], table)
    assert [len(item) for item in flows] == [1, 2]
    assert keys[0] == [("a", "c")]
    graphs[0].metadata["window_start_ns"] += 120 * 1_000_000_000
    with pytest.raises(ValueError, match="no flow records"):
        flows_for(graphs, table)
