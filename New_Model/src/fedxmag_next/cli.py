"""Minimal explicit CLI for a newly registered experimental protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .artifact import load_detector, save_detector
from .contracts import load_windows, require_normal
from .detector import ConditionalSupportDetector, DetectorConfig
from .model import CausalEdgeGraphEncoder, EncoderConfig
from .training import TrainingConfig, train_encoder


def _train(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    reference = require_normal(load_windows(args.train), "train/reference")
    selection = require_normal(load_windows(args.selection), "selection")
    calibration = require_normal(load_windows(args.calibration), "calibration")
    dimensions = (reference[0].node_dim, reference[0].edge_dim)
    if any((item.node_dim, item.edge_dim) != dimensions for item in [*selection, *calibration]):
        raise ValueError("all roles must use the same graph feature schema")
    encoder = CausalEdgeGraphEncoder(
        EncoderConfig(dimensions[0], dimensions[1], hidden_dim=args.hidden_dim, layers=args.layers)
    )
    history = train_encoder(
        encoder,
        reference,
        TrainingConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed),
    )
    detector = ConditionalSupportDetector.fit(
        encoder,
        reference,
        selection,
        calibration,
        config=DetectorConfig(
            tail_count=args.tail_count,
            max_components=args.max_components,
            min_component_size=args.min_component_size,
        ),
        provenance={
            "train": str(Path(args.train).resolve()),
            "selection": str(Path(args.selection).resolve()),
            "calibration": str(Path(args.calibration).resolve()),
            "training_history": history,
            "warning": "No scientific result: use only a preregistered, unconsumed protocol.",
        },
    )
    save_detector(args.artifact, detector)
    print(
        json.dumps(
            {"artifact": str(Path(args.artifact)), "thresholds": detector.thresholds}, indent=2
        )
    )


def _score(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    detector = load_detector(args.artifact)
    windows = load_windows(args.input)
    rows = detector.prediction_rows(windows)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {destination}")
    destination.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"predictions": str(destination), "count": len(rows)}, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train only from vetted normal roles")
    train.add_argument("--train", required=True)
    train.add_argument("--selection", required=True)
    train.add_argument("--calibration", required=True)
    train.add_argument("--artifact", required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--hidden-dim", type=int, default=64)
    train.add_argument("--layers", type=int, default=3)
    train.add_argument("--tail-count", type=int, default=3)
    train.add_argument("--max-components", type=int, default=8)
    train.add_argument("--min-component-size", type=int, default=32)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument(
        "--workers", type=int, default=6, help="CPU torch threads (validated sweet spot)"
    )
    train.set_defaults(run=_train)
    score = commands.add_parser("score", help="read-only scoring; does not self-adapt")
    score.add_argument("--artifact", required=True)
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    score.add_argument(
        "--workers", type=int, default=6, help="CPU torch threads (validated sweet spot)"
    )
    score.set_defaults(run=_score)
    return result


def main() -> None:
    args = parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
