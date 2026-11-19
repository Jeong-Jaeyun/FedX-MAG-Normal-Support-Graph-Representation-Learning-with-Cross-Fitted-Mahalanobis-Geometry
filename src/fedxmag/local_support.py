"""Train-normal nearest-neighbour and local-density novelty scores (exploratory).

LOF follows the reachability-distance definition, using training neighbours'
k-distances. Queries never join the reference set. Float64 torch avoids the
local environment's incompatible numpy BLAS runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fedxmag.m1.covariance import estimate_covariance


@dataclass
class LocalSupport:
    reference: Tensor
    k: int
    reference_k_distance: Tensor
    reference_lrd: Tensor
    mean: Tensor | None = None
    factor: Tensor | None = None

    @staticmethod
    def neighbours(
        query: Tensor,
        reference: Tensor,
        k: int,
        *,
        self_reference: bool = False,
        chunk_size: int = 256,
    ) -> tuple[Tensor, Tensor]:
        distances, indices = [], []
        for start in range(0, len(query), chunk_size):
            matrix = torch.cdist(query[start : start + chunk_size], reference)
            if self_reference:
                row = torch.arange(matrix.shape[0])
                matrix[row, row + start] = torch.inf
            values, neighbours = torch.topk(matrix, k, dim=1, largest=False, sorted=True)
            distances.append(values)
            indices.append(neighbours)
        return torch.cat(distances), torch.cat(indices)

    @classmethod
    @torch.no_grad()
    def fit(cls, reference: Tensor, *, k: int = 20, whiten: bool = False) -> LocalSupport:
        if reference.ndim != 2 or not torch.isfinite(reference).all():
            raise ValueError("Reference must be a finite matrix")
        if not 1 <= k < len(reference):
            raise ValueError("k must be positive and smaller than the reference size")
        values = reference.detach().cpu().double()
        mean = factor = None
        if whiten:
            estimate = estimate_covariance(values, "ledoit_wolf")
            mean = estimate.mean
            factor = torch.linalg.cholesky(estimate.covariance)
            values = torch.linalg.solve_triangular(factor, (values - mean).T, upper=False).T
        distances, indices = cls.neighbours(values, values, k, self_reference=True)
        k_distance = distances[:, -1]
        reachability = torch.maximum(distances, k_distance[indices])
        lrd = 1.0 / reachability.mean(dim=1).clamp_min(1e-10)
        return cls(values, k, k_distance, lrd, mean, factor)

    @torch.no_grad()
    def score(self, query: Tensor) -> dict[str, Tensor]:
        values = query.detach().cpu().double()
        if values.ndim != 2 or values.shape[1] != self.reference.shape[1]:
            raise ValueError("Query dimensions differ from reference")
        if not torch.isfinite(values).all():
            raise ValueError("Queries must be finite")
        if not len(values):
            return {key: values.new_empty(0) for key in ("knn", "lof")}
        if self.factor is not None:
            values = torch.linalg.solve_triangular(
                self.factor,
                (values - self.mean).T,
                upper=False,
            ).T
        distances, indices = self.neighbours(values, self.reference, self.k)
        reachability = torch.maximum(distances, self.reference_k_distance[indices])
        lrd = 1.0 / reachability.mean(dim=1).clamp_min(1e-10)
        return {
            "knn": distances.mean(dim=1),
            "lof": (self.reference_lrd[indices] / lrd[:, None]).mean(dim=1),
        }
