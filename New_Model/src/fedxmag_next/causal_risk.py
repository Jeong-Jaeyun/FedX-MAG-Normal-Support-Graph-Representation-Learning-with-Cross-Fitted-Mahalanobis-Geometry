"""Carry decaying prior alarm evidence only to the currently active same key.

This is deterministic keyed state around a frozen DNN, not a trained temporal
GNN. Only actual base-alarm contributors create risk. Verified normal feedback
clears current keys AFTER prediction; absent confirmation alone creates nothing.
"""

from __future__ import annotations

import math
from collections import OrderedDict, deque

import numpy as np


def scan_contributors(values, reference, aggregation="rank_scan", *, reference_sorted=False):
    if not len(values):
        return np.empty(0, dtype=np.int64)
    if aggregation == "top3":
        return np.argsort(-values, kind="stable")[:3]
    if aggregation != "rank_scan":
        raise ValueError("Unsupported base alarm aggregation")
    normal = reference if reference_sorted else np.sort(reference)
    order = np.argsort(-values, kind="stable")
    p = (1 + len(normal) - np.searchsorted(normal, values[order], side="left")) / (len(normal) + 1)
    statistic = np.log(np.arange(1, len(values) + 1) / len(values)) - np.log(p)
    count = int(np.argmax(statistic)) + 1
    return order[:count]


class CausalPeakMemory:
    def __init__(self, center, *, horizon=300, decay=300, maximum_keys=5000):
        if horizon <= 0 or decay <= 0 or maximum_keys <= 0:
            raise ValueError("Causal risk budgets must be positive")
        self.center = float(center)
        self.horizon, self.decay, self.maximum_keys = horizon, decay, maximum_keys
        self.states = OrderedDict()
        self.last_time, self.pending = None, None
        self.maximum_seen = 0

    def state_dict(self):
        if self.pending is not None:
            raise ValueError("Checkpoint risk only after prediction feedback is committed")
        return {
            "format_version": 1,
            "center": self.center,
            "horizon": self.horizon,
            "decay": self.decay,
            "maximum_keys": self.maximum_keys,
            "last_time": self.last_time,
            "maximum_seen": self.maximum_seen,
            "states": [(key, list(values)) for key, values in self.states.items()],
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("format_version") != 1:
            raise ValueError("Unsupported causal risk checkpoint")
        memory = cls(
            state["center"],
            horizon=state["horizon"],
            decay=state["decay"],
            maximum_keys=state["maximum_keys"],
        )
        memory.states = OrderedDict((key, deque(values)) for key, values in state["states"])
        if len(memory.states) > memory.maximum_keys:
            raise ValueError("Risk checkpoint exceeds its state budget")
        memory.last_time, memory.maximum_seen = state["last_time"], state["maximum_seen"]
        return memory

    def _peak(self, key, now):
        values = self.states.get(key)
        if values is None:
            return None
        while values and values[0][0] < now - self.horizon:
            values.popleft()
        if not values:
            del self.states[key]
            return None
        when, excess = values[0]
        return self.center + excess * math.exp(-(now - when) / self.decay)

    def predict(self, now, keys, values, observed_at):
        if self.pending is not None or (self.last_time is not None and now <= self.last_time):
            raise ValueError("Finish feedback before the next strictly later risk query")
        keys = list(keys)
        values = np.asarray(values, dtype=float)
        observed_at = np.asarray(observed_at, dtype=np.int64)
        if len(keys) != len(values) or len(values) != len(observed_at):
            raise ValueError("Risk keys, scores, and times must align")
        if not np.isfinite(values).all() or np.any(observed_at > now):
            raise ValueError("Risk query cannot contain nonfinite scores or future observations")
        if self.last_time is not None and np.any(observed_at < self.last_time):
            raise ValueError("Current risk evidence predates the previous completed query")
        peaks = {key: self._peak(key, now) for key in dict.fromkeys(keys)}
        output = values.copy()
        for i, key in enumerate(keys):
            if peaks[key] is not None:
                output[i] = max(output[i], peaks[key])
        self.pending = (now, keys, values.copy(), observed_at.copy())
        return output

    def feedback(self, *, verified_normal, contributors):
        if self.pending is None:
            raise ValueError("Normal feedback must follow a prediction")
        now, keys, values, times = self.pending
        if verified_normal:
            for key in set(keys):
                self.states.pop(key, None)
        else:
            contributors = np.asarray(contributors, dtype=np.int64)
            if np.any(contributors < 0) or np.any(contributors >= len(values)):
                raise ValueError("Risk contributor does not belong to the current window")
            ordered = contributors[np.argsort(times[contributors], kind="stable")]
            for i in ordered:
                excess = float(values[i] - self.center)
                if excess <= 0:
                    continue
                key, when = keys[i], int(times[i])
                state = self.states.setdefault(key, deque())
                while state and state[0][0] < now - self.horizon:
                    state.popleft()
                if state and state[-1][0] == when and state[-1][1] >= excess:
                    continue
                while (
                    state and state[-1][1] * math.exp(-(when - state[-1][0]) / self.decay) <= excess
                ):
                    state.pop()
                state.append((when, excess))
                self.states.move_to_end(key)
                while len(self.states) > self.maximum_keys:
                    self.states.popitem(last=False)
        self.maximum_seen = max(self.maximum_seen, len(self.states))
        self.last_time = now
        self.pending = None
