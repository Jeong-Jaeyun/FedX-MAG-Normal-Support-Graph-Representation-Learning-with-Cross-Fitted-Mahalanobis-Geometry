"""Bounded late fusion using symmetric ranks of heterogeneous window scores."""

from __future__ import annotations

from collections import deque

import numpy as np

from .joint_rank import joint_rank_pvalue


class RankConsensus:
    def __init__(self, calibration, capacity, pooling):
        values = np.asarray(calibration, dtype=float)
        if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
            raise ValueError("Consensus calibration must be a finite matrix")
        if capacity < 19 or pooling not in ("any", "mean", "median", "all"):
            raise ValueError("Consensus requires a supported .05 calibration budget and rule")
        self.dimension, self.capacity, self.pooling = values.shape[1], capacity, pooling
        self.history = deque(values[-capacity:].tolist(), maxlen=capacity)
        self.last_minute, self.pending = None, None

    def query(self, minute, scores, observed=True):
        if self.pending is not None or (
            self.last_minute is not None and minute <= self.last_minute
        ):
            raise ValueError("Commit feedback before a strictly later consensus query")
        values = self._vector(scores)
        valid = bool(observed and len(self.history) >= 19)
        pvalue = (
            joint_rank_pvalue(np.asarray(self.history), values, pooling=self.pooling)
            if valid
            else 1.0
        )
        self.pending = (int(minute), values.copy(), bool(observed))
        return bool(valid and pvalue <= 0.05), valid, pvalue

    def feedback(self, *, verified_normal, scores=None):
        if self.pending is None:
            raise ValueError("Consensus feedback requires a pending prediction")
        minute, query, observed = self.pending
        values = query if scores is None else self._vector(scores)
        if verified_normal and observed:
            self.history.append(values.tolist())
        self.last_minute, self.pending = minute, None

    def _vector(self, scores):
        values = np.asarray(scores, dtype=float)
        if values.shape != (self.dimension,) or not np.isfinite(values).all():
            raise ValueError("Consensus query and feedback must match the finite model schema")
        return values

    def state_dict(self):
        if self.pending is not None:
            raise ValueError("Commit consensus feedback before checkpointing")
        return {
            "format_version": 1,
            "capacity": self.capacity,
            "pooling": self.pooling,
            "history": list(self.history),
            "last_minute": self.last_minute,
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("format_version") != 1:
            raise ValueError("Unsupported heterogeneous consensus state")
        result = cls(state["history"], state["capacity"], state["pooling"])
        result.last_minute = state["last_minute"]
        return result
