"""Normal conditional density: current evidence given strictly past context.

Unlike joint anomaly energy, this cancels the marginal surprise of the history.
The joint Gaussian mixture is fitted on source normals only. Conditioning uses
Cholesky blocks and mixture posterior weights determined by context alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .geometry import ConditionalMahalanobis


@dataclass(frozen=True)
class HistoryConditionalSupport:
    joint: ConditionalMahalanobis
    context_dimension: int

    @classmethod
    def fit(cls, current: Tensor, context: Tensor, modes: int = 1):
        if current.ndim != 2 or context.ndim != 2 or len(current) != len(context):
            raise ValueError("Expected aligned current and context matrices")
        if current.shape[1] == 0 or context.shape[1] == 0:
            raise ValueError("Both current evidence and history are required")
        joint = ConditionalMahalanobis.fit(
            torch.cat((context, current), dim=1), max_components=modes
        )
        return cls(joint, context.shape[1])

    def score(self, current: Tensor, context: Tensor) -> Tensor:
        dimension = self.context_dimension
        if (
            current.ndim != 2
            or context.ndim != 2
            or len(current) != len(context)
            or context.shape[1] != dimension
            or current.shape[1] != self.joint.dimension - dimension
        ):
            raise ValueError("Conditional query schema differs from fitted support")
        if not torch.isfinite(current).all() or not torch.isfinite(context).all():
            raise ValueError("Conditional queries must be finite")
        factor = self.joint.cholesky
        history_factor = factor[:, :dimension, :dimension]
        cross_factor = factor[:, dimension:, :dimension]
        residual_factor = factor[:, dimension:, dimension:]
        history_delta = (context.double()[:, None] - self.joint.means[:, :dimension]).permute(
            1, 2, 0
        )
        history_white = torch.linalg.solve_triangular(history_factor, history_delta, upper=False)
        history_cost = history_white.square().sum(dim=1)
        history_logdet_half = history_factor.diagonal(dim1=1, dim2=2).log().sum(dim=1)
        log_weights = (
            self.joint.log_prior[:, None] - 0.5 * history_cost - history_logdet_half[:, None]
        )
        log_weights = log_weights - torch.logsumexp(log_weights, dim=0)
        current_delta = (current.double()[:, None] - self.joint.means[:, dimension:]).permute(
            1, 2, 0
        )
        residual = current_delta - cross_factor @ history_white
        residual_white = torch.linalg.solve_triangular(residual_factor, residual, upper=False)
        residual_cost = residual_white.square().sum(dim=1)
        residual_logdet_half = residual_factor.diagonal(dim1=1, dim2=2).log().sum(dim=1)
        # Omit only the constant current_dimension * log(2*pi), shared by all rows.
        return -2 * torch.logsumexp(
            log_weights - 0.5 * residual_cost - residual_logdet_half[:, None], dim=0
        )

    def state_dict(self) -> dict:
        return {"joint": self.joint.state_dict(), "context_dimension": self.context_dimension}

    def marginal_score(self, current: Tensor) -> Tensor:
        """Score p(current) of the SAME joint model, for a matched conditioning control."""
        dimension = self.context_dimension
        factor = self.joint.cholesky[:, dimension:, :]
        covariance = factor @ factor.transpose(1, 2)
        marginal_factor = torch.linalg.cholesky(covariance)
        delta = (current.double()[:, None] - self.joint.means[:, dimension:]).permute(1, 2, 0)
        whitened = torch.linalg.solve_triangular(marginal_factor, delta, upper=False)
        logdet_half = marginal_factor.diagonal(dim1=1, dim2=2).log().sum(dim=1)
        log_density = self.joint.log_prior[:, None] - 0.5 * whitened.square().sum(dim=1)
        return -2 * torch.logsumexp(log_density - logdet_half[:, None], dim=0)

    @classmethod
    def from_state_dict(cls, state: dict):
        return cls(
            ConditionalMahalanobis.from_state_dict(state["joint"]), state["context_dimension"]
        )
