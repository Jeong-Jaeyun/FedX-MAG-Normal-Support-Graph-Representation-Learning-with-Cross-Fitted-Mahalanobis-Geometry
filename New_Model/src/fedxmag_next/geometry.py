"""Multimodal shrinkage-Mahalanobis normal support."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _ledoit_wolf_alpha(centered: Tensor) -> Tensor:
    """Stable isotropic shrinkage intensity for a centered reference matrix."""

    samples, dimensions = centered.shape
    covariance = centered.T @ centered / samples
    mu = torch.trace(covariance) / dimensions
    beta = (centered.square().T @ centered.square()).sum() / samples**2
    delta = covariance.square().sum() - 2 * mu * torch.trace(covariance) + dimensions * mu.square()
    alpha = (beta - covariance.square().sum() / samples).clamp_min(0.0)
    return (alpha / delta.clamp_min(torch.finfo(centered.dtype).eps)).clamp(0.0, 1.0)


def _shrinkage_covariance(values: Tensor, floor: float) -> tuple[Tensor, Tensor]:
    """Return mean and rotation-equivariant, positive-definite covariance."""

    values = values.double()
    mean = values.mean(dim=0)
    centered = values - mean
    dimensions = values.shape[1]
    empirical = centered.T @ centered / max(values.shape[0] - 1, 1)
    scale = (torch.trace(empirical) / dimensions).clamp_min(floor)
    alpha = _ledoit_wolf_alpha(centered)
    covariance = (1.0 - alpha) * empirical + alpha * scale * torch.eye(
        dimensions, dtype=values.dtype, device=values.device
    )
    covariance = covariance + floor * scale * torch.eye(
        dimensions, dtype=values.dtype, device=values.device
    )
    return mean, covariance


def _kmeans(values: Tensor, clusters: int, steps: int, seed: int) -> Tensor:
    """Small deterministic k-means used only on normal-reference embeddings."""

    values = values.double()
    generator = torch.Generator(device=values.device).manual_seed(seed)
    initial = torch.randperm(values.shape[0], generator=generator, device=values.device)[:clusters]
    centers = values[initial].clone()
    assignments = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
    for _ in range(steps):
        new_assignments = torch.cdist(values, centers).argmin(dim=1)
        next_centers = []
        for component in range(clusters):
            rows = values[new_assignments == component]
            next_centers.append(rows.mean(dim=0) if len(rows) else centers[component])
        next_centers_tensor = torch.stack(next_centers)
        if torch.equal(new_assignments, assignments) and torch.allclose(
            next_centers_tensor, centers
        ):
            assignments = new_assignments
            break
        assignments, centers = new_assignments, next_centers_tensor
    return assignments


@dataclass(frozen=True)
class ConditionalMahalanobis:
    """A normal-support mixture; components are fitted from trusted normal data."""

    means: Tensor
    cholesky: Tensor
    log_prior: Tensor
    counts: Tensor

    @property
    def components(self) -> int:
        return int(self.means.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])

    @classmethod
    def fit(
        cls,
        reference: Tensor,
        *,
        max_components: int = 8,
        min_component_size: int = 32,
        kmeans_steps: int = 40,
        seed: int = 17,
        covariance_floor: float = 1e-5,
    ) -> ConditionalMahalanobis:
        if reference.ndim != 2 or not torch.isfinite(reference).all():
            raise ValueError("Reference must be a finite [entities, dimensions] tensor")
        if len(reference) < max(2, min_component_size):
            raise ValueError("Too few reference entities for a stable normal-support component")
        # A component cannot have enough samples otherwise.  Empty/small clusters
        # are merged into their closest retained normal mode below.
        proposed = min(max_components, max(1, len(reference) // min_component_size))
        assignments = _kmeans(reference, proposed, kmeans_steps, seed)
        centers = []
        kept = []
        for component in range(proposed):
            rows = reference[assignments == component]
            if len(rows) >= min_component_size:
                centers.append(rows.mean(dim=0))
                kept.append(component)
        if not kept:
            kept, centers = [0], [reference.mean(dim=0)]
            assignments = torch.zeros(len(reference), dtype=torch.long, device=reference.device)
        else:
            # Reassign every point to a retained mode before final statistics.
            assignments = torch.cdist(reference.double(), torch.stack(centers).double()).argmin(
                dim=1
            )
        means, factors, counts = [], [], []
        for component in range(len(centers)):
            rows = reference[assignments == component]
            if len(rows) < 2:
                # This can only arise from a degenerate final reassignment; merge
                # into the largest retained group rather than inventing a mode.
                continue
            mean, covariance = _shrinkage_covariance(rows, covariance_floor)
            means.append(mean)
            factors.append(torch.linalg.cholesky(covariance))
            counts.append(len(rows))
        if not means:
            raise RuntimeError("No viable normal-support component was fitted")
        count_tensor = torch.tensor(counts, dtype=torch.float64, device=reference.device)
        return cls(
            torch.stack(means),
            torch.stack(factors),
            (count_tensor / count_tensor.sum()).log(),
            count_tensor.to(torch.long),
        )

    def component_energies(self, queries: Tensor) -> Tensor:
        if queries.ndim != 2 or queries.shape[1] != self.dimension:
            raise ValueError("Query embedding dimension differs from normal support")
        # One factor per component, with all queries as right-hand sides. This
        # avoids duplicating a d-by-d factor for every query entity.
        difference = (queries.double()[:, None, :] - self.means[None, :, :]).permute(1, 2, 0)
        solved = torch.linalg.solve_triangular(self.cholesky, difference, upper=False)
        return solved.square().sum(dim=1).T

    def score(self, queries: Tensor) -> tuple[Tensor, Tensor]:
        """Prior-adjusted minimum entity energy and its supporting normal mode."""

        components = self.component_energies(queries) - 2.0 * self.log_prior
        return components.min(dim=1)

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "means": self.means.cpu(),
            "cholesky": self.cholesky.cpu(),
            "log_prior": self.log_prior.cpu(),
            "counts": self.counts.cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor]) -> ConditionalMahalanobis:
        return cls(state["means"], state["cholesky"], state["log_prior"], state["counts"])
