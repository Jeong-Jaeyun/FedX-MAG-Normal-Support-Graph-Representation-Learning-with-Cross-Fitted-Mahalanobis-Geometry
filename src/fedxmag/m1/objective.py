"""Cross-fitted normal-support objective and covariance regularizers (M1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from fedxmag.m1.covariance import (
    ESTIMATORS,
    differentiable_shape_terms,
    estimate_covariance,
    mahalanobis_energy,
)

ENERGY_MODES = ("cross_fitted", "in_sample")


@dataclass(frozen=True)
class NormalSupportSettings:
    """Frozen M1 hyperparameters; ``lambda_m == 0`` reduces M1 to M0."""

    estimator: str = "ledoit_wolf"
    energy_mode: str = "cross_fitted"
    symmetric: bool = True
    fixed_alpha: float = 0.5
    spectral_floor: float = 1e-6
    lambda_m: float = 1.0
    lambda_c: float = 1.0
    lambda_s: float = 1.0
    beta_kappa: float = 1.0
    beta_rank: float = 1.0
    kappa_max: float = 100.0
    rank_min: float = 8.0
    scale_min: float = 0.01
    scale_max: float = 100.0
    reconstruction_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"Unknown estimator: {self.estimator}")
        if self.energy_mode not in ENERGY_MODES:
            raise ValueError(f"Unknown energy mode: {self.energy_mode}")
        for name in ("lambda_m", "lambda_c", "lambda_s", "reconstruction_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0.0 <= self.fixed_alpha <= 1.0:
            raise ValueError("fixed_alpha must be in [0, 1]")
        if not 0 < self.scale_min < self.scale_max:
            raise ValueError("Expected 0 < scale_min < scale_max")

    @property
    def uses_normal_support(self) -> bool:
        return self.lambda_m > 0 or self.lambda_c > 0 or self.lambda_s > 0


@dataclass
class NormalSupportLoss:
    total: Tensor
    energy: Tensor
    shape: Tensor
    scale: Tensor
    diagnostics: dict[str, float] = field(default_factory=dict)


def _regularizers(estimate: Any, settings: NormalSupportSettings) -> tuple[Tensor, Tensor]:
    log_condition, effective_rank = differentiable_shape_terms(estimate.shape)
    shape = settings.beta_kappa * torch.relu(
        log_condition - torch.log(torch.tensor(settings.kappa_max, dtype=torch.float64))
    ).square() + settings.beta_rank * torch.relu(settings.rank_min - effective_rank).square()
    scale = torch.relu(settings.scale_min - estimate.scale).square() + torch.relu(
        estimate.scale - settings.scale_max
    ).square()
    return shape, scale


def normal_support_loss(
    embeddings: Tensor,
    settings: NormalSupportSettings,
    generator: torch.Generator,
) -> NormalSupportLoss:
    """Energy normalized by ``d`` plus shape and scale guardrails.

    In ``cross_fitted`` mode the batch is split into disjoint reference and query
    halves; only queries are scored (optionally symmetrized by swapping roles).
    ``in_sample`` mode fits and scores the same rows and exists only as the
    self-evaluation-bias ablation.
    """

    count, dimension = embeddings.shape
    estimator_kwargs = {
        "fixed_alpha": settings.fixed_alpha,
        "spectral_floor": settings.spectral_floor,
    }
    pairs: list[tuple[Tensor, Tensor]] = []
    if settings.energy_mode == "in_sample":
        pairs.append((embeddings, embeddings))
    else:
        if count < 4:
            raise ValueError("Cross-fitting needs at least four embeddings")
        order = torch.randperm(count, generator=generator).to(embeddings.device)
        half = count // 2
        reference, query = embeddings[order[:half]], embeddings[order[half : 2 * half]]
        pairs.append((reference, query))
        if settings.symmetric:
            pairs.append((query, reference))

    energies: list[Tensor] = []
    shapes: list[Tensor] = []
    scales: list[Tensor] = []
    alphas: list[float] = []
    reference_scales: list[float] = []
    for reference, query in pairs:
        estimate = estimate_covariance(reference, settings.estimator, **estimator_kwargs)
        energies.append(mahalanobis_energy(query, estimate).mean() / dimension)
        shape_term, scale_term = _regularizers(estimate, settings)
        shapes.append(shape_term)
        scales.append(scale_term)
        alphas.append(float(estimate.alpha))
        reference_scales.append(float(estimate.scale.detach()))
    energy = torch.stack(energies).mean()
    shape = torch.stack(shapes).mean()
    scale = torch.stack(scales).mean()
    total = settings.lambda_m * energy + settings.lambda_c * shape + settings.lambda_s * scale
    return NormalSupportLoss(
        total=total,
        energy=energy,
        shape=shape,
        scale=scale,
        diagnostics={
            "alpha": sum(alphas) / len(alphas),
            "scale": sum(reference_scales) / len(reference_scales),
            "energy_over_d": float(energy.detach()),
            "shape_penalty": float(shape.detach()),
            "scale_penalty": float(scale.detach()),
        },
    )
