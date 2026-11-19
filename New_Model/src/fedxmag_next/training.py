"""Normal-only masked-reconstruction training for the new encoder."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from .contracts import GraphWindow, MaskedBatch, collate, mask_features, require_normal
from .model import CausalEdgeGraphEncoder


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    mask_rate: float = 0.25
    gradient_clip_norm: float = 5.0
    seed: int = 17

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("invalid training configuration")
        if not 0 < self.mask_rate < 1:
            raise ValueError("mask_rate must lie in (0, 1)")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if not mask.any():
        return prediction.sum() * 0.0
    return (prediction[mask] - target[mask]).square().mean()


def reconstruction_loss(output_node: Tensor, output_edge: Tensor, masked: MaskedBatch) -> Tensor:
    """Only artificial, naturally available corruptions are reconstruction targets."""

    return _masked_mse(output_node, masked.node_target, masked.node_mask) + _masked_mse(
        output_edge, masked.edge_target, masked.edge_mask
    )


def train_encoder(
    encoder: CausalEdgeGraphEncoder,
    windows: Iterable[GraphWindow],
    config: TrainingConfig | None = None,
    *,
    verbose: bool = False,
) -> list[dict[str, float]]:
    """Train reconstruction only; no Mahalanobis energy is in this objective."""

    config = config or TrainingConfig()
    items = require_normal(windows, "encoder training")
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    device = next(encoder.parameters()).device
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        encoder.train()
        order = torch.randperm(len(items), generator=generator).tolist()
        total = 0.0
        batches = 0
        for start in range(0, len(order), config.batch_size):
            chosen = [items[index] for index in order[start : start + config.batch_size]]
            raw = collate(chosen).to(device)
            masked = mask_features(raw, config.mask_rate, generator)
            output = encoder(masked)
            loss = reconstruction_loss(output.node_prediction, output.edge_prediction, masked)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.gradient_clip_norm)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        history.append({"epoch": float(epoch), "masked_reconstruction_loss": total / batches})
        if verbose:
            print(history[-1], flush=True)
    encoder.eval()
    return history
