"""Deterministic, synthetic software-path check.  It is not detection evidence."""
# ruff: noqa: E402  # Local source-path bootstrap when run directly from the repository.

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from fedxmag_next.contracts import GraphWindow
from fedxmag_next.detector import ConditionalSupportDetector, DetectorConfig
from fedxmag_next.model import CausalEdgeGraphEncoder, EncoderConfig
from fedxmag_next.training import TrainingConfig, train_encoder


def make_window(seed: int, shift: float = 0.0) -> GraphWindow:
    generator = torch.Generator().manual_seed(seed)
    nodes, edges = 6, 8
    index = torch.stack((torch.arange(edges) % nodes, (torch.arange(edges) + 1) % nodes))
    return GraphWindow(
        torch.randn(nodes, 4, generator=generator) + shift,
        torch.randn(edges, 3, generator=generator) + shift,
        index.long(),
        torch.ones(nodes, 4, dtype=torch.bool),
        torch.ones(edges, 3, dtype=torch.bool),
        0,
        {"session_key": f"synthetic-{seed}", "window_start_ns": seed},
    )


def main() -> None:
    windows = [make_window(index) for index in range(140)]
    encoder = CausalEdgeGraphEncoder(EncoderConfig(4, 3, hidden_dim=12, layers=2))
    train_encoder(encoder, windows[:80], TrainingConfig(epochs=2, batch_size=8))
    detector = ConditionalSupportDetector.fit(
        encoder,
        windows[:80],
        windows[80:110],
        windows[110:140],
        config=DetectorConfig(max_components=3, min_component_size=16),
    )
    score = detector.score_window(make_window(999, shift=4.0))
    assert torch.isfinite(score.node_energies).all() and torch.isfinite(score.edge_energies).all()
    assert detector.thresholds["0.05"] is not None
    print("Synthetic path passed; it is not a scientific performance result.")


if __name__ == "__main__":
    main()
