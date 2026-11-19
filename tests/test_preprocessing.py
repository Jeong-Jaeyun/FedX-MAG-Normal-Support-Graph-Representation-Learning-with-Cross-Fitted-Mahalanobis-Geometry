from pathlib import Path

import numpy as np
import pytest
import torch

from fedxmag.preflight import _reciprocal_timing_exact_match
from fedxmag.preprocessing import (
    CanonicalEvent,
    FeatureStandardizer,
    build_static_graphs,
    chronological_window_aligned_split,
    read_xiiotid_events,
)
from fedxmag.schema import (
    DEFAULT_FEATURE_GROUPS,
    RESOLUTION_AWARE_FEATURE_GROUPS,
    RESOLUTION_AWARE_V3A_FEATURE_GROUPS,
    GraphSample,
)
from fedxmag.serialization import load_graph_samples, save_graph_samples


def _event(
    second: int,
    size: float | None,
    attack: bool = False,
    session: str = "capture-a",
) -> CanonicalEvent:
    return CanonicalEvent(
        timestamp_ns=second * 1_000_000_000,
        src_key="device-a",
        dst_key="gateway",
        directional_mean_packet_size=size,
        protocol="MQTT",
        is_attack=attack,
        session_key=session,
    )


def test_graph_builder_has_exact_feature_order_and_availability() -> None:
    graph = build_static_graphs([_event(1, 10), _event(2, 20), _event(3, 50)])[0]
    assert graph.node_features.shape == (2, 14)
    assert graph.edge_features.shape == (1, 7)
    assert graph.edge_index.tolist() == [[0], [1]]
    assert graph.edge_available[0].tolist() == [True] * 7
    assert np.isclose(float(graph.edge_features[0, 0]), np.log1p(1000.0))
    assert np.isclose(float(graph.edge_features[0, 1]), 0.0)
    assert np.isclose(float(graph.edge_features[0, 2]), -1.0)
    assert np.isclose(float(graph.edge_features[0, 6]), np.log1p(44.0))


def test_resolution_aware_timing_separates_spacing_and_multiplicity() -> None:
    events = [_event(second, 10.0) for second in (1, 1, 3, 6)]
    graph = build_static_graphs(
        events,
        groups=RESOLUTION_AWARE_V3A_FEATURE_GROUPS,
        timing_profile="distinct_second_multiplicity_v3",
        timestamp_resolution_seconds=1,
    )[0]
    timing = graph.edge_features[0, :5]
    availability = graph.edge_available[0, :5]

    assert availability.tolist() == [True] * 5
    assert np.isclose(float(timing[0]), np.log1p(2.5))
    assert np.isclose(float(timing[1]), np.log1p(0.5))
    assert np.isclose(float(timing[2]), np.log1p(2.95))
    assert np.isclose(float(timing[3]), 0.25)
    assert np.isclose(float(timing[4]), np.log1p(1.9))


def test_resolution_aware_timing_does_not_treat_same_second_as_zero_iat() -> None:
    graph = build_static_graphs(
        [_event(1, 10.0), _event(1, 20.0), _event(1, 30.0)],
        groups=RESOLUTION_AWARE_V3A_FEATURE_GROUPS,
        timing_profile="distinct_second_multiplicity_v3",
        timestamp_resolution_seconds=1,
    )[0]
    timing = graph.edge_features[0, :5]
    availability = graph.edge_available[0, :5]

    assert availability.tolist() == [False, False, False, True, True]
    assert torch.equal(timing[:3], torch.zeros(3))
    assert np.isclose(float(timing[3]), 2.0 / 3.0)
    assert np.isclose(float(timing[4]), np.log1p(3.0))


def test_final_resolution_aware_profile_drops_redundant_multiplicity_target() -> None:
    graph = build_static_graphs(
        [_event(1, 10.0), _event(1, 20.0), _event(3, 30.0)],
        groups=RESOLUTION_AWARE_FEATURE_GROUPS,
        timing_profile="distinct_second_collision_v3",
        timestamp_resolution_seconds=1,
    )[0]

    assert graph.edge_features.shape[1] == RESOLUTION_AWARE_FEATURE_GROUPS.edge_dim == 7
    assert graph.node_features.shape[1] == RESOLUTION_AWARE_FEATURE_GROUPS.node_dim == 14
    assert graph.edge_available[0, :4].tolist() == [True, False, False, True]
    assert np.isclose(float(graph.edge_features[0, 3]), 1.0 / 3.0)


def test_window_aligned_split_prevents_shared_windows() -> None:
    events = [_event(second, 20.0) for second in range(0, 240, 10)]
    splits = chronological_window_aligned_split(
        events,
        {"train": 0.5, "select": 0.25, "test": 0.25},
        window_seconds=60,
    )
    split_windows = {
        name: {event.timestamp_ns // 60_000_000_000 for event in selected}
        for name, selected in splits.items()
    }
    assert split_windows["train"].isdisjoint(split_windows["select"])
    assert split_windows["train"].isdisjoint(split_windows["test"])
    assert split_windows["select"].isdisjoint(split_windows["test"])


def test_xiiotid_adapter_and_train_only_scaler(tmp_path: Path) -> None:
    csv_path = tmp_path / "tiny.csv"
    csv_path.write_text(
        "Timestamp,Scr_IP,Des_IP,Scr_bytes,Des_bytes,Scr_pkts,Des_pkts,Protocol,class1\n"
        "1,a,b,100,60,10,3,MQTT,Normal\n"
        "2,a,b,150,80,10,4,MQTT,Normal\n"
        "3,a,b,200,100,10,5,MQTT,Normal\n",
        encoding="utf-8",
    )
    result = read_xiiotid_events(csv_path)
    assert len(result.events) == 6
    assert result.report.reverse_events == 3
    assert result.report.timing_observed_events == 3
    assert result.report.synthetic_reverse_events_excluded_from_timing == 3
    assert [event.timing_observed for event in result.events].count(True) == 3
    assert [event.timing_observed for event in result.events].count(False) == 3
    assert result.report.label_values == {"normal": 3}
    assert all(not event.is_attack for event in result.events)
    graphs = build_static_graphs(result.events, window_seconds=60)
    scaler = FeatureStandardizer.fit(graphs)
    transformed = scaler.transform_many(graphs)
    assert transformed[0].label == 0
    assert np.isfinite(transformed[0].node_features.numpy()).all()
    artifact = tmp_path / "graphs.pt"
    save_graph_samples(artifact, transformed)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        save_graph_samples(artifact, transformed)
    restored = load_graph_samples(artifact)
    assert len(restored) == 1
    assert restored[0].edge_index.tolist() == transformed[0].edge_index.tolist()
    assert restored[0].metadata == transformed[0].metadata


def test_synthetic_reverse_observations_are_distribution_only(tmp_path: Path) -> None:
    csv_path = tmp_path / "reciprocal.csv"
    csv_path.write_text(
        "Timestamp,Scr_IP,Des_IP,Scr_bytes,Des_bytes,Scr_pkts,Des_pkts,class1\n"
        "1,a,b,100,60,10,3,Normal\n"
        "2,a,b,120,80,10,4,Normal\n"
        "3,a,b,140,100,10,5,Normal\n",
        encoding="utf-8",
    )
    result = read_xiiotid_events(csv_path)
    graph = build_static_graphs(result.events, window_seconds=60)[0]
    edge_lookup = {
        tuple(nodes): index
        for index, nodes in enumerate(graph.metadata["edge_keys"])
    }
    forward = edge_lookup[("a", "b")]
    reverse = edge_lookup[("b", "a")]
    timing = list(DEFAULT_FEATURE_GROUPS.indices("edge", "timing"))
    distribution = list(DEFAULT_FEATURE_GROUPS.indices("edge", "distribution"))

    assert graph.edge_available[forward, timing].all()
    assert not graph.edge_available[reverse, timing].any()
    assert graph.edge_available[reverse, distribution].all()
    assert graph.metadata["timing_observed_event_count"] == 3
    assert graph.metadata["synthetic_reverse_event_count"] == 3
    diagnostic = _reciprocal_timing_exact_match([graph])
    assert diagnostic["reciprocal_edge_pairs"] == 1
    assert diagnostic["jointly_available_timing_scalars"] == 0
    assert diagnostic["reciprocal_exact_match_rate"] == 0.0
    assert diagnostic["reciprocal_exact_match_exposure_rate"] == 0.0


def test_missing_counters_keep_timing_and_do_not_create_missing_nodes(tmp_path: Path) -> None:
    csv_path = tmp_path / "missing.csv"
    csv_path.write_text(
        "Timestamp,Scr_IP,Des_IP,Scr_bytes,Des_bytes,Scr_pkts,Des_pkts,class1\n"
        "1,a,b,-,?,10,?,Normal\n"
        "2,?,b,100,20,10,2,Normal\n",
        encoding="utf-8",
    )
    result = read_xiiotid_events(csv_path)
    assert len(result.events) == 1
    assert result.events[0].directional_mean_packet_size is None
    assert result.report.rows_skipped_missing_endpoint == 1
    assert result.report.size_unavailable_events == 1
    assert all(event.src_key != "?" and event.dst_key != "?" for event in result.events)


def test_sessions_with_same_timestamp_build_separate_graphs() -> None:
    graphs = build_static_graphs(
        [_event(1, 10.0, session="one"), _event(1, 20.0, session="two")]
    )
    assert len(graphs) == 2
    assert {graph.metadata["session_key"] for graph in graphs} == {"one", "two"}


def test_within_session_split_keeps_complete_windows_disjoint() -> None:
    events = [
        _event(second, 20.0, session=session)
        for session in ("one", "two")
        for second in range(0, 600, 60)
    ]
    splits = chronological_window_aligned_split(
        events,
        {
            "train": 0.50,
            "select": 0.10,
            "audit": 0.10,
            "calibration": 0.15,
            "test": 0.15,
        },
        window_seconds=60,
        session_split_strategy="within_session_chronological",
    )
    identities = {
        name: {(event.session_key, event.timestamp_ns // 60_000_000_000) for event in selected}
        for name, selected in splits.items()
    }
    assert all(
        identities[left].isdisjoint(identities[right])
        for index, left in enumerate(identities)
        for right in list(identities)[index + 1 :]
    )
    assert all(
        {event.session_key for event in selected} == {"one", "two"}
        for selected in splits.values()
    )


def test_unknown_label_is_not_silently_treated_as_attack(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad-label.csv"
    csv_path.write_text(
        "Timestamp,Scr_IP,Des_IP,class1\n1,a,b,unknown\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown binary label"):
        read_xiiotid_events(csv_path)


def test_released_xiiotid_class_hierarchy_is_mapped_correctly(tmp_path: Path) -> None:
    csv_path = tmp_path / "released-schema.csv"
    csv_path.write_text(
        "Date,Timestamp,Scr_IP,Des_IP,class1,class2,class3\n"
        "9/01/2020,1,a,b,Scanning_vulnerability,Reconnaissance,Attack\n",
        encoding="utf-8",
    )
    result = read_xiiotid_events(csv_path)
    assert len(result.events) == 1
    event = result.events[0]
    assert event.is_attack
    assert event.attack_category == "Reconnaissance"
    assert event.attack_subcategory == "Scanning_vulnerability"
    assert event.session_key.endswith(":9/01/2020")


def test_exact_duplicate_rows_are_reported_and_removed(tmp_path: Path) -> None:
    csv_path = tmp_path / "duplicates.csv"
    header = "Timestamp,Scr_IP,Des_IP,class1\n"
    row = "1,a,b,Normal\n"
    csv_path.write_text(header + row + row, encoding="utf-8")
    result = read_xiiotid_events(csv_path)
    assert len(result.events) == 1
    assert result.report.rows_total == 2
    assert result.report.exact_duplicate_rows_detected == 1
    assert result.report.rows_skipped_exact_duplicate == 1
    assert result.report.raw_label_values == {"normal": 2}
    assert result.report.label_values == {"normal": 1}


def test_standardizer_gives_each_graph_equal_weight() -> None:
    groups = DEFAULT_FEATURE_GROUPS

    def sample(entities: int, value: float) -> GraphSample:
        return GraphSample(
            node_features=torch.full((entities, groups.node_dim), value),
            edge_features=torch.full((entities, groups.edge_dim), value),
            edge_index=torch.stack(
                (torch.arange(entities), torch.arange(entities)), dim=0
            ),
            node_available=torch.ones((entities, groups.node_dim), dtype=torch.bool),
            edge_available=torch.ones((entities, groups.edge_dim), dtype=torch.bool),
            label=0,
        )

    scaler = FeatureStandardizer.fit([sample(1, 0.0), sample(10, 10.0)])
    assert torch.allclose(scaler.node_mean, torch.full((groups.node_dim,), 5.0))
    assert torch.allclose(scaler.edge_mean, torch.full((groups.edge_dim,), 5.0))
