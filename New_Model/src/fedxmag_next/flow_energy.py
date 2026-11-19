"""Invertible deep Gaussian-energy mapping (Real NVP affine couplings).

The exact Jacobian correction prevents arbitrary normal-energy contraction from
becoming a free improvement in the training loss. No input dimension is removed
by the DNN. Constant training dimensions are retained as explicit support checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


class AffineCoupling(nn.Module):
    def __init__(self, dimension: int, hidden: int, parity: int) -> None:
        super().__init__()
        self.register_buffer("mask", ((torch.arange(dimension) + parity) % 2).float())
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * dimension),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def parameters_for(self, fixed: Tensor) -> tuple[Tensor, Tensor]:
        scale, shift = self.network(fixed).chunk(2, dim=1)
        return 1.5 * torch.tanh(scale) * (1 - self.mask), shift * (1 - self.mask)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        fixed = values * self.mask
        scale, shift = self.parameters_for(fixed)
        return fixed + (1 - self.mask) * (values * torch.exp(scale) + shift), scale.sum(dim=1)

    def inverse(self, latent: Tensor) -> Tensor:
        fixed = latent * self.mask
        scale, shift = self.parameters_for(fixed)
        return fixed + (1 - self.mask) * ((latent - shift) * torch.exp(-scale))


@dataclass(frozen=True)
class FlowConfig:
    dimension: int
    hidden: int = 64
    layers: int = 6


class InvertibleEnergy(nn.Module):
    def __init__(self, config: FlowConfig) -> None:
        super().__init__()
        self.config = config
        self.couplings = nn.ModuleList(
            AffineCoupling(config.dimension, config.hidden, layer % 2)
            for layer in range(config.layers)
        )

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        latent, logdet = values, values.new_zeros(len(values))
        for layer in self.couplings:
            latent, increment = layer(latent)
            logdet = logdet + increment
        return latent, logdet

    def inverse(self, latent: Tensor) -> Tensor:
        values = latent
        for layer in reversed(self.couplings):
            values = layer.inverse(values)
        return values

    def scores(self, values: Tensor) -> dict[str, Tensor]:
        latent, logdet = self(values)
        energy = 0.5 * latent.square().sum(dim=1)
        return {
            "energy": energy,
            "nll": energy - logdet + self.config.dimension * math.log(2 * math.pi) / 2,
        }


def train_flow(
    model: InvertibleEnergy,
    train: Tensor,
    selection: Tensor,
    *,
    epochs: int = 40,
    seed: int = 17,
    batch_size: int = 256,
) -> list[dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        total, count = 0.0, 0
        for indices in order.split(batch_size):
            # Small fixed dequantization noise avoids singular densities from
            # discrete protocol/share features. It is never attack-dependent.
            values = train[indices] + 0.02 * torch.randn(train[indices].shape, generator=generator)
            nll = model.scores(values)["nll"].mean()
            if not torch.isfinite(nll):
                raise RuntimeError("Nonfinite normalizing-flow loss")
            optimizer.zero_grad(set_to_none=True)
            nll.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(nll.detach()) * len(indices)
            count += len(indices)
        model.eval()
        with torch.no_grad():
            validation = float(model.scores(selection)["nll"].mean())
        history.append(
            {"epoch": epoch, "train_nll": total / count, "selection_normal_nll": validation}
        )
        if validation < best_loss:
            best_loss = validation
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(history[-1], flush=True)
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    return history
