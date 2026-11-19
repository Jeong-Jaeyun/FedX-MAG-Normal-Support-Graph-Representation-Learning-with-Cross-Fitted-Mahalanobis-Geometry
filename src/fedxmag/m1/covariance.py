"""Covariance estimators, Mahalanobis energies, and spectral diagnostics for M1.

All matrix algebra runs in float64 torch. The shrinkage target is the scaled
identity ``s I`` with ``s = tr(S) / d`` (rotation-equivariant isotropic
shrinkage, formulation v0.3 section 7).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

ESTIMATORS = ("empirical", "fixed", "ledoit_wolf", "oas")
DEFAULT_SPECTRAL_FLOOR = 1e-6


@dataclass
class CovarianceEstimate:
    mean: Tensor
    covariance: Tensor
    shape: Tensor
    scale: Tensor
    alpha: Tensor
    sample_covariance: Tensor


def ledoit_wolf_shrinkage(centered: Tensor) -> Tensor:
    """Ledoit-Wolf (2004) intensity toward ``mu I`` for already-centered rows.

    Uses the biased sample covariance, matching the reference formulation.
    """

    samples, features = centered.shape
    squared = centered.square()
    variance_trace = squared.sum(dim=0) / samples
    mu = variance_trace.sum() / features
    beta_sum = (squared.T @ squared).sum()
    delta_sum = (centered.T @ centered).square().sum() / samples**2
    beta = (beta_sum / samples - delta_sum) / (features * samples)
    delta = (delta_sum - 2.0 * mu * variance_trace.sum() + features * mu**2) / features
    beta = torch.minimum(beta, delta)
    return (beta / delta.clamp_min(torch.finfo(delta.dtype).tiny)).clamp(0.0, 1.0)


def oas_shrinkage(centered: Tensor) -> Tensor:
    """Oracle Approximating Shrinkage (Chen et al., 2010; corrected form)."""

    samples, features = centered.shape
    covariance = centered.T @ centered / samples
    alpha = covariance.square().mean()
    mu = torch.trace(covariance) / features
    numerator = alpha + mu**2
    denominator = (samples + 1) * (alpha - mu**2 / features)
    if float(denominator) <= 0.0:
        return torch.ones((), dtype=centered.dtype, device=centered.device)
    return (numerator / denominator).clamp(0.0, 1.0)


def estimate_covariance(
    embeddings: Tensor,
    estimator: str,
    *,
    fixed_alpha: float = 0.5,
    spectral_floor: float = DEFAULT_SPECTRAL_FLOOR,
    stop_gradient_alpha: bool = True,
) -> CovarianceEstimate:
    """Return ``Sigma = s [(1 - alpha) S / s + alpha I + floor I]``.

    The floor is a numerical-safety term on the trace-normalized shape only; it
    is not a scale identification device.
    """

    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown covariance estimator: {estimator}")
    values = embeddings.to(torch.float64)
    samples, features = values.shape
    if samples < 2:
        raise ValueError("At least two reference embeddings are required")
    mean = values.mean(dim=0)
    centered = values - mean
    sample_covariance = centered.T @ centered / (samples - 1)
    # Clamping only guards an exactly constant reference set.
    scale = (torch.trace(sample_covariance) / features).clamp_min(1e-12)
    shape = sample_covariance / scale
    shrinkage_input = centered.detach() if stop_gradient_alpha else centered
    if estimator == "empirical":
        alpha = values.new_zeros(())
    elif estimator == "fixed":
        alpha = values.new_tensor(float(fixed_alpha))
    elif estimator == "ledoit_wolf":
        alpha = ledoit_wolf_shrinkage(shrinkage_input)
    else:
        alpha = oas_shrinkage(shrinkage_input)
    if stop_gradient_alpha:
        alpha = alpha.detach()
    identity = torch.eye(features, dtype=values.dtype, device=values.device)
    shrunk_shape = (1.0 - alpha) * shape + alpha * identity + spectral_floor * identity
    return CovarianceEstimate(
        mean=mean,
        covariance=scale * shrunk_shape,
        shape=shrunk_shape,
        scale=scale,
        alpha=alpha,
        sample_covariance=sample_covariance,
    )


def mahalanobis_energy(queries: Tensor, estimate: CovarianceEstimate) -> Tensor:
    """``(z - mu)^T Sigma^{-1} (z - mu)`` for every query row."""

    difference = (queries.to(torch.float64) - estimate.mean).T
    factor = torch.linalg.cholesky(estimate.covariance)
    solved = torch.linalg.solve_triangular(factor, difference, upper=False)
    return solved.square().sum(dim=0)


def spectral_diagnostics(matrix: Tensor) -> dict[str, float]:
    """Spectrum summaries of a covariance or trace-normalized shape matrix."""

    eigenvalues = torch.linalg.eigvalsh(matrix.detach().to(torch.float64))
    positive = eigenvalues.clamp_min(0.0)
    total = positive.sum()
    probabilities = positive / total.clamp_min(torch.finfo(total.dtype).tiny)
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum()
    return {
        "lambda_min": float(eigenvalues[0]),
        "lambda_max": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0])
        if float(eigenvalues[0]) > 0
        else float("inf"),
        "effective_rank_entropy": float(entropy.exp()),
        "participation_ratio": float(total.square() / positive.square().sum()),
        "trace": float(torch.trace(matrix.detach().to(torch.float64))),
    }


def differentiable_shape_terms(shape: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(log condition number, entropy effective rank)`` with gradients."""

    eigenvalues = torch.linalg.eigvalsh(shape).clamp_min(1e-12)
    log_condition = eigenvalues[-1].log() - eigenvalues[0].log()
    probabilities = eigenvalues / eigenvalues.sum()
    effective_rank = (-(probabilities * probabilities.log()).sum()).exp()
    return log_condition, effective_rank


def linear_cka(left: Tensor, right: Tensor) -> float:
    """Linear centered kernel alignment between two row-aligned embeddings."""

    x = left.to(torch.float64) - left.to(torch.float64).mean(dim=0)
    y = right.to(torch.float64) - right.to(torch.float64).mean(dim=0)
    cross = (x.T @ y).square().sum()
    norm = ((x.T @ x).square().sum() * (y.T @ y).square().sum()).sqrt()
    return float(cross / norm) if float(norm) > 0 else float("nan")
