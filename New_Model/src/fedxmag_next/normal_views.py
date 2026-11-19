"""Marginal current-flow normal geometry, leaving the anomaly encoder intact."""

from __future__ import annotations

import torch

from .daily_detector import DailyDeepEnergyDetector
from .deep_neighbors import whiten


class CurrentNormalProjector:
    def __init__(self, base, *, projection_state=None):
        self.base = DailyDeepEnergyDetector(base)
        if projection_state is None:
            columns = [
                i for i, name in enumerate(base["feature_names"]) if not name.startswith("context_")
            ]
            if not columns or len(columns) == len(base["feature_names"]):
                raise ValueError(
                    "Current-flow normal view requires both current and past-context fields"
                )
            original = base["raw_supports"]["raw_mahalanobis_m1"]
            if original["means"].shape[0] != 1:
                raise ValueError("Current-flow normal view needs the frozen source1-mode metric")
            covariance = original["cholesky"][0] @ original["cholesky"][0].T
            marginal = covariance[columns][:, columns]
            self.state = {
                "format_version": 1,
                "columns": columns,
                "raw_geometry": original,
                "current_geometry": {
                    **original,
                    "means": original["means"][:, columns].clone(),
                    "cholesky": torch.linalg.cholesky(marginal)[None],
                },
            }
        else:
            if projection_state.get("format_version") != 1:
                raise ValueError("Unsupported current-flow normal projection state")
            self.state = projection_state

    @torch.no_grad()
    def project(self, features):
        if features.ndim != 2 or features.shape[1] != len(self.base.feature_names):
            raise ValueError("Current-flow normal projection feature schema differs")
        if not torch.isfinite(features).all():
            raise ValueError("Current-flow normal projection needs finite observed features")
        normalized = self.base.normalize_features(features).double()
        return {
            "raw": whiten(normalized, self.state["raw_geometry"]),
            "current": whiten(normalized[:, self.state["columns"]], self.state["current_geometry"]),
        }
