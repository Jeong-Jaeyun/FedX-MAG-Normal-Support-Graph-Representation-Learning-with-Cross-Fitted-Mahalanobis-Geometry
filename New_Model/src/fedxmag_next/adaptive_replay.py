"""Recalibrate a newly available model using strictly prior reserved normals."""

from __future__ import annotations

from collections import deque

import numpy as np

from .commission_roles import latest_ready_stage
from .detector import conformal_threshold


def replay_adaptive(
    minutes, labels, observed, base_energy, base_calibration, stages, *, capacity=39, fpr=0.05
):
    n = len(minutes)
    flags, valid = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    energy, threshold, pvalue = np.zeros(n), np.full(n, np.nan), np.ones(n)
    assignment = np.zeros(n, dtype=np.int64)
    ready = [stage["ready_at"] for stage in stages]
    if len(ready) and not np.all(np.diff(ready) > 0):
        raise ValueError("Adaptive models must become available in chronological order")
    history = deque(base_calibration[-capacity:], maxlen=capacity)
    current = 0
    for i, minute in enumerate(minutes):
        active = latest_ready_stage((int(minute) + 1) * 60, ready)
        if active != current:
            stage = stages[active - 1]
            history = deque(stage["calibration"][-capacity:], maxlen=capacity)
            # Re-score only completed, verified normal windows reserved for
            # this new model. No training/selection window can enter its buffer.
            for j in range(i):
                if labels[j] == 0 and observed[j] and stage["admit"][j]:
                    history.append(stage["energy"][j])
            current = active
        values = stages[current - 1]["energy"] if current else base_energy
        allowed = stages[current - 1]["admit"][i] if current else True
        value = float(values[i])
        cutoff = conformal_threshold(history, fpr)
        energy[i], assignment[i] = value, current
        valid[i] = bool(observed[i] and cutoff is not None)
        if valid[i]:
            threshold[i] = cutoff
            flags[i] = value > cutoff
            pvalue[i] = (1 + np.sum(np.asarray(history) >= value)) / (1 + len(history))
        # The normal feedback is used AFTER this window's decision.
        if labels[i] == 0 and observed[i] and allowed:
            history.append(value)
    return {
        "flags": flags,
        "valid": valid,
        "energy": energy,
        "threshold": threshold,
        "pvalue": pvalue,
        "model_assignment": assignment,
    }
