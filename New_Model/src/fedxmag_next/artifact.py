"""Versioned persistence for the new detector artifacts."""

from __future__ import annotations

from pathlib import Path

import torch

from .detector import ConditionalSupportDetector


def save_detector(path: str | Path, detector: ConditionalSupportDetector) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite detector artifact: {destination}")
    torch.save(detector.state_dict(), destination)


def load_detector(path: str | Path) -> ConditionalSupportDetector:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("Detector artifact is not a dictionary")
    return ConditionalSupportDetector.from_state_dict(state)
