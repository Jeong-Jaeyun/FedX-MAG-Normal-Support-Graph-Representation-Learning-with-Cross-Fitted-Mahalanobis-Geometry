"""Normal-only learned transformations, adapted from NeuTraL AD (ICML2021).

Qiu et al., https://proceedings.mlr.press/v139/qiu21a.html, equations2/3.
This local controlled variant uses four sigmoid masks, a64-wide SiLU encoder,
unit-length embeddings, and the same30-epoch/three-seed source budget as controls.
Masks act on standardized feature coordinates, not synthetic wire-level flows.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class UnitNorm(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        return F.normalize(values, dim=-1, eps=1e-6)


def deterministic_contrastive_score(embedded: Tensor, temperature: float) -> Tensor:
    """Sum transformed-view losses, excluding each view's self-similarity."""
    if embedded.ndim != 3 or embedded.shape[1] < 3 or temperature <= 0:
        raise ValueError("Expected original plus at least two views and positive temperature")
    normalized = F.normalize(embedded, dim=-1, eps=1e-6)
    logits = normalized[:, 1:] @ normalized.transpose(1, 2) / temperature
    views = embedded.shape[1] - 1
    own_view = torch.zeros(views, views + 1, dtype=torch.bool, device=embedded.device)
    own_view[torch.arange(views), torch.arange(views) + 1] = True
    partition = torch.logsumexp(logits.masked_fill(own_view, -torch.inf), dim=-1)
    return (partition - logits[:, :, 0]).sum(dim=1)


class TransformationEnergy(nn.Module):
    def __init__(
        self, dimension: int, transforms: int = 4, temperature: float = 0.1, hidden: int = 64
    ):
        super().__init__()
        if transforms < 2 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                "At least two transformations and positive finite temperature required"
            )
        self.dimension, self.temperature = dimension, temperature
        # Retain the shared layers[:-1] encoder interface used by node pooling.
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            UnitNorm(),
            nn.Identity(),
        )
        self.masks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dimension, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, dimension),
                    nn.Sigmoid(),
                )
                for _ in range(transforms)
            ]
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError("Transformation input schema differs")
        views = torch.stack([values, *(values * mask(values) for mask in self.masks)], dim=1)
        embedded = self.layers(views.flatten(0, 1)).unflatten(0, views.shape[:2])
        return deterministic_contrastive_score(embedded, self.temperature)


def train_transformations(model, reference, selection, *, seed, epochs):
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    history, best_state, best_value = [], None, float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for indices in torch.randperm(len(reference), generator=generator).split(256):
            loss = model(reference[indices]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite normal transformation objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            count += len(indices)
        model.eval()
        with torch.no_grad():
            validation = float(torch.cat([model(x) for x in selection.split(8192)]).mean())
        entry = {"epoch": epoch, "train_dcl": total / count, "selection_normal_dcl": validation}
        history.append(entry)
        if validation < best_value:
            best_value = validation
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0:
            print(entry, flush=True)
    if best_state is None:
        raise ValueError("No finite transformation checkpoint selected")
    model.load_state_dict(best_state)
    model.eval()
    return history
