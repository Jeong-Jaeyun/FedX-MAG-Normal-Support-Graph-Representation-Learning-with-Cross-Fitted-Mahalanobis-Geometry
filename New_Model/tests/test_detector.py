from __future__ import annotations

import pytest
import torch

from fedxmag_next.artifact import load_detector, save_detector
from fedxmag_next.contracts import GraphWindow
from fedxmag_next.detector import (
    CausalStateStore,
    ConditionalSupportDetector,
    DetectorConfig,
    conformal_threshold,
    top_tail,
)
from fedxmag_next.model import CausalEdgeGraphEncoder, EncoderConfig
from fedxmag_next.training import TrainingConfig, train_encoder


def window(index: int, label: int = 0, shift: float = 0.0) -> GraphWindow:
    generator = torch.Generator().manual_seed(index)
    node_count, edge_count = 5, 6
    return GraphWindow(
        torch.randn(node_count, 3, generator=generator) + shift,
        torch.randn(edge_count, 2, generator=generator) + shift,
        torch.stack(
            (torch.arange(edge_count) % node_count, (torch.arange(edge_count) + 1) % node_count)
        ),
        torch.ones(node_count, 3, dtype=torch.bool),
        torch.ones(edge_count, 2, dtype=torch.bool),
        label,
        {
            "session_key": f"session-{index}",
            "window_start_ns": index,
            "endpoint_keys": [f"node-{item}" for item in range(node_count)],
            "edge_keys": [
                [f"node-{item % node_count}", f"node-{(item + 1) % node_count}"]
                for item in range(edge_count)
            ],
        },
    )


@pytest.fixture(scope="module")
def detector() -> ConditionalSupportDetector:
    items = [window(index) for index in range(130)]
    encoder = CausalEdgeGraphEncoder(EncoderConfig(3, 2, hidden_dim=10, layers=2))
    train_encoder(encoder, items[:70], TrainingConfig(epochs=1, batch_size=10))
    return ConditionalSupportDetector.fit(
        encoder,
        items[:70],
        items[70:100],
        items[100:130],
        config=DetectorConfig(max_components=3, min_component_size=16),
    )


def test_tail_is_not_graph_mean() -> None:
    assert top_tail(torch.tensor([0.0, 0.0, 9.0]), 1).item() == 9.0
    assert top_tail(torch.tensor([0.0, 0.0, 9.0]), 2).item() == 4.5


def test_five_percent_threshold_abstains_below_nineteen_vetted_windows() -> None:
    assert conformal_threshold(range(18), 0.05) is None
    assert conformal_threshold(range(19), 0.05) is not None


def test_detector_scores_and_has_calibration_only_threshold(
    detector: ConditionalSupportDetector,
) -> None:
    result = detector.score_window(window(999, shift=3.0))
    assert result.score > 0
    assert detector.thresholds["0.05"] is not None
    assert len(result.node_energies) == 5
    assert len(result.edge_energies) == 6


def test_reference_roles_cannot_overlap(detector: ConditionalSupportDetector) -> None:
    shared = [window(index) for index in range(40)]
    with pytest.raises(ValueError, match="overlap"):
        ConditionalSupportDetector.fit(detector.encoder, shared[:20], shared[10:30], shared[30:])


def test_causal_state_cannot_admit_attack(detector: ConditionalSupportDetector) -> None:
    state = CausalStateStore(detector.encoder_config.hidden_dim)
    with pytest.raises(ValueError, match="vetted normal"):
        detector.admit_vetted_normal_to_context(window(700, label=1), state)
    normal = window(701)
    detector.admit_vetted_normal_to_context(normal, state)
    before = {key: value.clone() for key, value in state.nodes.items()}
    detector.score_window(window(702, label=1), state)
    assert set(state.nodes) == set(before)
    assert all(torch.equal(state.nodes[key], value) for key, value in before.items())


def test_tensor_only_artifact_round_trip(
    detector: ConditionalSupportDetector, tmp_path: pytest.TempPathFactory
) -> None:
    path = tmp_path / "detector.pt"
    save_detector(path, detector)
    recovered = load_detector(path)
    original = detector.score_window(window(999)).score
    restored = recovered.score_window(window(999)).score
    assert restored == pytest.approx(original)
