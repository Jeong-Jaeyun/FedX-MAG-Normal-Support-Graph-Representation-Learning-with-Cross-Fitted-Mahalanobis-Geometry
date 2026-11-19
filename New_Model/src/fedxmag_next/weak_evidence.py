"""Bounded per-window original evidence, with peak or additive decay."""

from __future__ import annotations

import math
from collections import OrderedDict, deque

import numpy as np


class CausalEvidenceMemory:
    def __init__(self, center, mode, horizon=300, decay=300, maximum_keys=5000):
        if mode not in ("peak", "sum") or min(horizon, decay, maximum_keys) <= 0:
            raise ValueError("Invalid temporal evidence mode or positive budgets")
        self.center, self.mode = float(center), mode
        self.horizon, self.decay, self.maximum_keys = horizon, decay, maximum_keys
        self.states = OrderedDict()
        self.last_time, self.pending = None, None
        self.maximum_seen = self.maximum_entries = 0

    def predict(self, now, keys, values, times):
        values, times = np.asarray(values, dtype=float), np.asarray(times, dtype=np.int64)
        if self.pending is not None or (self.last_time is not None and now <= self.last_time):
            raise ValueError("Commit evidence before a strictly later prediction")
        if len(keys) != len(values) or times.shape != values.shape or not np.isfinite(values).all():
            raise ValueError("Temporal evidence metadata must align with finite original scores")
        if np.any(times >= now) or (self.last_time is not None and np.any(times < self.last_time)):
            raise ValueError("Current evidence leaves its completed observation interval")
        prior = {}
        for key in dict.fromkeys(keys):
            events = self.states.get(key)
            if events is None:
                continue
            while events and events[0][0] < now - self.horizon:
                events.popleft()
            if not events:
                del self.states[key]
                continue
            excess = [value * math.exp(-(now - when) / self.decay) for when, value in events]
            prior[key] = self.center + (max(excess) if self.mode == "peak" else sum(excess))
        predicted = np.array(
            [max(value, prior.get(key, value)) for key, value in zip(keys, values, strict=True)]
        )
        self.pending = (int(now), list(keys), values.copy(), times.copy())
        return predicted

    def commit(self, contributors):
        if self.pending is None:
            raise ValueError("Evidence must follow a prediction")
        now, keys, values, times = self.pending
        best = {}
        for i in np.asarray(contributors, dtype=np.int64):
            if i < 0 or i >= len(values):
                raise ValueError("Contributor is outside its original observation window")
            excess = float(values[i] - self.center)
            if excess <= 0:
                continue
            when, key = int(times[i]), keys[i]
            decayed = excess * math.exp(-(now - when) / self.decay)
            if key not in best or decayed > best[key][0]:
                best[key] = (decayed, when, excess)
        for key, (_, when, excess) in best.items():
            events = self.states.setdefault(key, deque())
            while events and events[0][0] < now - self.horizon:
                events.popleft()
            events.append((when, excess))
            self.states.move_to_end(key)
            while len(self.states) > self.maximum_keys:
                self.states.popitem(last=False)
        self.last_time, self.pending = now, None
        self.maximum_seen = max(self.maximum_seen, len(self.states))
        self.maximum_entries = max(self.maximum_entries, sum(len(v) for v in self.states.values()))

    def state_dict(self):
        if self.pending is not None:
            raise ValueError("Checkpoint only committed original evidence")
        return {
            "format_version": 1,
            "center": self.center,
            "mode": self.mode,
            "horizon": self.horizon,
            "decay": self.decay,
            "maximum_keys": self.maximum_keys,
            "last_time": self.last_time,
            "states": [(k, list(v)) for k, v in self.states.items()],
            "maximum_seen": self.maximum_seen,
            "maximum_entries": self.maximum_entries,
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("format_version") != 1:
            raise ValueError("Unsupported original-evidence checkpoint")
        memory = cls(
            *(state[key] for key in ("center", "mode", "horizon", "decay", "maximum_keys"))
        )
        memory.states = OrderedDict((key, deque(value)) for key, value in state["states"])
        if len(memory.states) > memory.maximum_keys:
            raise ValueError("Original-evidence checkpoint exceeds its key budget")
        memory.last_time = state["last_time"]
        memory.maximum_seen, memory.maximum_entries = (
            state["maximum_seen"],
            state["maximum_entries"],
        )
        return memory
