from dataclasses import replace

import numpy as np
import pytest
import torch

from fedxmag.flow_context import ABSOLUTE_SCALE_COLUMNS, FLOW_DIM, FLOW_VARIANTS
from fedxmag.fusion_detector import RawFlowDetector, TrafficWindows, time_blocks
from fedxmag.local_support import LocalSupport
from fedxmag.m1.detection import Standardizer, conformal_threshold
from fedxmag.m1.entity import aggregate, raw_entity_features
from fedxmag.schema import DEFAULT_FEATURE_GROUPS
from fedxmag.synthetic import make_synthetic_graphs

GROUPS = DEFAULT_FEATURE_GROUPS


def windows(count: int, seed: int, *, shift: float = 0.0) -> TrafficWindows:
    graphs = make_synthetic_graphs(num_graphs=count, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    flows = []
    for index, graph in enumerate(graphs):
        graph.node_features = graph.node_features + shift * graph.node_available
        graph.metadata.update(session_key=f"capture-{seed}", window_start_ns=index * 60)
        size = (graph.edge_features.shape[0], FLOW_DIM)
        flows.append(torch.randn(size, generator=generator, dtype=torch.float64) + shift)
    return TrafficWindows(graphs, flows)


def channel(model: LocalSupport, rows: list[torch.Tensor]) -> np.ndarray:
    scores = model.score(torch.cat(rows))["knn"]
    return aggregate(scores.split([len(item) for item in rows]), "top3_mean")


def manual_scores(train: TrafficWindows, selection: TrafficWindows, queries: TrafficWindows):
    scaler = Standardizer.fit(torch.cat(list(train.flows)))
    node = LocalSupport.fit(torch.cat(raw_entity_features(train.graphs).node), k=20)
    flow = LocalSupport.fit(scaler(torch.cat(list(train.flows))), k=20)

    def channels(items: TrafficWindows) -> tuple[np.ndarray, np.ndarray]:
        return (
            channel(node, raw_entity_features(items.graphs).node),
            channel(flow, [scaler(item) for item in items.flows]),
        )

    scales = [float(np.quantile(values, 0.95)) for values in channels(selection)]
    raw, context = channels(queries)
    return np.maximum(raw / scales[0], context / scales[1])


@pytest.fixture
def source() -> RawFlowDetector:
    return RawFlowDetector.fit(
        GROUPS, windows(12, 11), windows(20, 13), windows(19, 17), provenance={"run": 7}
    )


def test_fit_matches_independent_knn_fusion_and_survives_serialization(source, tmp_path) -> None:
    queries = windows(5, 19)
    expected = manual_scores(windows(12, 11), windows(20, 13), queries)
    fused, channels, entities = source.score(queries)
    np.testing.assert_array_equal(fused, expected)
    assert set(channels) == set(entities) == {"raw_node", "flow"}
    calibration = source.score(windows(19, 17))[0]
    np.testing.assert_array_equal(source.calibration_scores, calibration)
    # 19 calibration windows support the 5% and 10% targets, but not 1%.
    assert source.thresholds["0.01"] is None
    assert source.thresholds["0.05"] == conformal_threshold(calibration, 0.95)
    assert source.environment == "source" and source.reference_sessions == ("capture-11",)
    assert source.calibration_sessions == ("capture-17",) and source.flow_variant == "all"
    path = tmp_path / "detector.pt"
    source.save(path)
    restored = RawFlowDetector.load(path)
    assert restored.thresholds == source.thresholds
    assert restored.provenance == {"run": 7}
    np.testing.assert_array_equal(restored.score(queries)[0], fused)
    with pytest.raises(FileExistsError):
        source.save(path)


def test_commissioning_extends_reference_with_fixed_scalers(source, tmp_path) -> None:
    reference, calibration = windows(10, 23, shift=3.0), windows(19, 29, shift=3.0)
    later = windows(6, 31, shift=3.0)
    before = source.score(later)[0]
    node_rows, flow_rows = len(source.node_reference), len(source.flow_reference)
    adapted = source.commission(
        reference, calibration, environment="site-b", record={"vetting": "ticket-1"}
    )
    # The source detector is untouched; scalers and fusion scales are reused.
    assert len(source.node_reference) == node_rows and source.environment == "source"
    np.testing.assert_array_equal(source.score(later)[0], before)
    assert adapted.scales == source.scales and adapted.flow_scaler is source.flow_scaler
    combined = torch.cat((source.node_reference, *raw_entity_features(reference.graphs).node))
    torch.testing.assert_close(adapted.node_reference, combined, rtol=0, atol=0)
    assert len(adapted.flow_reference) == flow_rows + sum(len(f) for f in reference.flows)
    expected = np.maximum(
        channel(LocalSupport.fit(combined, k=20), raw_entity_features(later.graphs).node)
        / source.scales["raw_node"],
        channel(
            LocalSupport.fit(adapted.flow_reference, k=20),
            [source.flow_scaler(item) for item in later.flows],
        )
        / source.scales["flow"],
    )
    np.testing.assert_array_equal(adapted.score(later)[0], expected)
    assert adapted.score(later)[0].mean() < before.mean()
    np.testing.assert_array_equal(adapted.calibration_scores, adapted.score(calibration)[0])
    assert adapted.environment == "site-b"
    assert adapted.reference_sessions == ("capture-11", "capture-23")
    assert adapted.calibration_sessions == ("capture-29",)
    assert [step["action"] for step in adapted.provenance["commissioning"]] == [
        "reference_added",
        "recalibrated",
    ]
    assert adapted.provenance["commissioning"][0]["vetting"] == "ticket-1"
    assert "commissioning" not in source.provenance
    adapted.save(tmp_path / "adapted.pt")
    restored = RawFlowDetector.load(tmp_path / "adapted.pt")
    assert restored.provenance == adapted.provenance and restored.environment == "site-b"
    np.testing.assert_array_equal(restored.score(later)[0], adapted.score(later)[0])
    # Threshold-only recalibration keeps the source ranking and reference.
    recalibrated = source.recalibrate(calibration, environment="site-b")
    np.testing.assert_array_equal(recalibrated.score(later)[0], before)
    assert recalibrated.reference_sessions == source.reference_sessions
    assert recalibrated.calibration_sessions == ("capture-29",)
    assert recalibrated.thresholds != source.thresholds


def fused_with(detector: RawFlowDetector, extra: TrafficWindows, queries: TrafficWindows):
    node = torch.cat((detector.node_reference, *raw_entity_features(extra.graphs).node))
    flow = torch.cat((detector.flow_reference, *(detector.flow_scaler(f) for f in extra.flows)))
    return np.maximum(
        channel(LocalSupport.fit(node, k=20), raw_entity_features(queries.graphs).node)
        / detector.scales["raw_node"],
        channel(LocalSupport.fit(flow, k=20), [detector.flow_scaler(f) for f in queries.flows])
        / detector.scales["flow"],
    )


def test_cross_fitted_commissioning_never_scores_a_block_against_itself(source) -> None:
    commissioning, later = windows(24, 53, shift=3.0), windows(6, 59, shift=3.0)
    # Two interleaved captures whose time order is the reverse of their list order.
    for index, graph in enumerate(commissioning.graphs):
        graph.metadata.update(session_key=f"capture-{index % 2}", window_start_ns=(23 - index) * 60)
    blocks = time_blocks(commissioning, 3)
    assert sorted(i for block in blocks for i in block) == list(range(24))
    for capture in (0, 1):
        members = sorted(
            (i for i in range(24) if i % 2 == capture),
            key=lambda i: commissioning.graphs[i].metadata["window_start_ns"],
        )
        assigned = [next(b for b, block in enumerate(blocks) if i in block) for i in members]
        assert assigned == [0] * 4 + [1] * 4 + [2] * 4
    adapted = source.commission(commissioning, environment="site-d", blocks=3)
    expected = np.empty(24)
    for block in blocks:
        others = commissioning.subset([i for i in range(24) if i not in block])
        expected[block] = fused_with(source, others, commissioning.subset(block))
    np.testing.assert_array_equal(adapted.calibration_scores, expected)
    np.testing.assert_array_equal(adapted.score(later)[0], fused_with(source, commissioning, later))
    # In-sample scores against the final reference can only be lower than held-out scores.
    assert (adapted.score(commissioning)[0] <= adapted.calibration_scores).all()
    assert adapted.scales == source.scales and adapted.environment == "site-d"
    assert adapted.calibration_sessions == ("capture-0", "capture-1")
    (step,) = adapted.provenance["commissioning"]
    assert step["action"] == "reference_added_cross_fitted_calibration"
    assert step["blocks"] == 3 and step["calibration_windows"] == 24
    assert "commissioning" not in source.provenance


def test_cross_fitted_commissioning_refuses_ambiguous_or_small_inputs(source) -> None:
    commissioning = windows(24, 61)
    with pytest.raises(ValueError, match="either"):
        source.commission(commissioning, environment="site-e")
    with pytest.raises(ValueError, match="either"):
        source.commission(commissioning, windows(19, 67), environment="site-e", blocks=2)
    with pytest.raises(ValueError, match="two time blocks"):
        source.commission(commissioning, environment="site-e", blocks=1)
    with pytest.raises(ValueError, match="19 calibration"):
        source.commission(commissioning.subset(range(18)), environment="site-e", blocks=2)
    del commissioning.graphs[0].metadata["window_start_ns"]
    with pytest.raises(ValueError, match="metadata"):
        source.commission(commissioning, environment="site-e", blocks=2)


def test_fit_and_commissioning_refuse_unsafe_inputs(source) -> None:
    reference, calibration = windows(10, 37), windows(19, 41)
    reference.graphs[0].label = 1
    with pytest.raises(ValueError, match="normal-only"):
        source.commission(reference, calibration, environment="site-c")
    reference.graphs[0].label = 0
    with pytest.raises(ValueError, match="disjoint"):
        source.commission(reference, reference, environment="site-c")
    with pytest.raises(ValueError, match="19 calibration"):
        source.commission(reference, windows(18, 43), environment="site-c")
    with pytest.raises(ValueError, match="20 selection"):
        RawFlowDetector.fit(GROUPS, windows(12, 11), windows(19, 13), windows(19, 17))
    with pytest.raises(ValueError, match="one flow-context"):
        TrafficWindows(reference.graphs, reference.flows[:-1])
    with pytest.raises(ValueError, match="Flow context"):
        TrafficWindows(reference.graphs[:1], [torch.zeros(2, FLOW_DIM - 1)])
    empty = source.score(TrafficWindows([], []))
    assert empty[0].size == 0


def test_predictions_localize_both_channels_and_ignore_labels(source) -> None:
    queries = windows(1, 47)
    sample, flows = queries.graphs[0], queries.flows[0]
    sample.metadata["endpoint_keys"] = [f"host-{i}" for i in range(sample.node_features.shape[0])]
    keys = [[f"src-{i}", f"dst-{i}"] for i in range(len(flows))]
    fused, channels, entities = source.score(queries)
    # Equality to the threshold is not an alert.
    source.calibration_scores[:] = fused[0]
    prediction = source.predict(queries, [keys])[0]
    assert prediction["score"] == fused[0]
    assert all(alert is False for alert in prediction["alerts"].values() if alert is not None)
    assert prediction["alerts"]["0.01"] is None
    assert prediction["capture_in_reference"] is False
    assert prediction["capture_in_calibration"] is False
    normalized = {name: channels[name][0] / source.scales[name] for name in channels}
    assert prediction["dominant_channel"] == max(normalized, key=normalized.get)
    highest = int(torch.argmax(entities["raw_node"][0]))
    assert prediction["top_nodes"][0]["endpoint_key"] == f"host-{highest}"
    flow_highest = int(torch.argmax(entities["flow"][0]))
    assert prediction["top_flows"][0]["edge_key"] == keys[flow_highest]
    attack = replace(sample, label=1, metadata={**sample.metadata, "attack_categories": ["DoS"]})
    assert source.predict(TrafficWindows([attack], [flows]), [keys]) == [prediction]
    with pytest.raises(ValueError, match="flow_keys"):
        source.predict(queries, [keys[:-1]])


def test_flow_variants_transform_each_window() -> None:
    flows = torch.arange(3 * FLOW_DIM, dtype=torch.float64).reshape(3, FLOW_DIM)
    flows[:, 0] = torch.tensor([1.0, 5.0, 2.0])
    relative = FLOW_VARIANTS["window_relative"](flows)
    assert relative[:, 0].tolist() == [-1.0, 3.0, 0.0]
    kept = [column for column in range(FLOW_DIM) if column not in ABSOLUTE_SCALE_COLUMNS]
    torch.testing.assert_close(relative[:, kept], flows[:, kept], rtol=0, atol=0)
    assert FLOW_VARIANTS["window_relative"](flows[:0]).shape == (0, FLOW_DIM)
    assert FLOW_VARIANTS["activity_flags"](flows).shape == (3, 9)
    assert FLOW_VARIANTS["composition"](flows).shape == (3, 12)
    assert FLOW_VARIANTS["all"](flows) is flows


@pytest.mark.parametrize("variant", ("activity_flags", "composition", "window_relative"))
def test_flow_variant_changes_only_the_flow_channel(source, variant, tmp_path) -> None:
    train, queries = windows(12, 11), windows(5, 19)
    detector = RawFlowDetector.fit(
        GROUPS, train, windows(20, 13), windows(19, 17), flow_variant=variant
    )
    transform = FLOW_VARIANTS[variant]
    reference = torch.cat([transform(item) for item in train.flows])
    scaler = Standardizer.fit(reference)
    expected = channel(
        LocalSupport.fit(scaler(reference), k=20), [scaler(transform(f)) for f in queries.flows]
    )
    _, channels, _ = detector.score(queries)
    np.testing.assert_array_equal(channels["flow"], expected)
    np.testing.assert_array_equal(channels["raw_node"], source.score(queries)[1]["raw_node"])
    adapted = detector.commission(windows(10, 23), windows(19, 29), environment="site-v")
    added = sum(len(item) for item in windows(10, 23).flows)
    assert len(adapted.flow_reference) == len(detector.flow_reference) + added
    detector.save(tmp_path / "variant.pt")
    restored = RawFlowDetector.load(tmp_path / "variant.pt")
    assert restored.flow_variant == variant
    np.testing.assert_array_equal(restored.score(queries)[0], detector.score(queries)[0])
    with pytest.raises(ValueError, match="Unknown flow variant"):
        RawFlowDetector.fit(GROUPS, train, windows(20, 13), windows(19, 17), flow_variant="x")


def test_version_one_artifacts_load_with_unknown_calibration_coverage(source, tmp_path) -> None:
    source.save(tmp_path / "current.pt")
    payload = torch.load(tmp_path / "current.pt", weights_only=True)
    payload["format_version"] = 1
    del payload["flow_variant"], payload["calibration_sessions"]
    torch.save(payload, tmp_path / "legacy.pt")
    legacy = RawFlowDetector.load(tmp_path / "legacy.pt")
    assert legacy.flow_variant == "all" and legacy.calibration_sessions == ()
    prediction = legacy.predict(windows(1, 71))[0]
    assert prediction["capture_in_calibration"] is None
    assert prediction["capture_in_reference"] is False


def test_load_rejects_tampered_artifacts(source, tmp_path) -> None:
    path = tmp_path / "detector.pt"
    source.save(path)
    payload = torch.load(path, weights_only=True)
    payload["thresholds"]["0.05"] += 1.0
    torch.save(payload, tmp_path / "bad_threshold.pt")
    with pytest.raises(ValueError, match="thresholds"):
        RawFlowDetector.load(tmp_path / "bad_threshold.pt")
    payload = torch.load(path, weights_only=True)
    payload["flow_features"][0] = "packets"
    torch.save(payload, tmp_path / "bad_flow.pt")
    with pytest.raises(ValueError, match="flow-context"):
        RawFlowDetector.load(tmp_path / "bad_flow.pt")
