"""Factorial prototype filtering and temporal evidence with shared frozen feedback."""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .causal_risk import CausalPeakMemory, scan_contributors
from .detector import conformal_threshold
from .prototype_guard import entity_keys
from .window_bridge import aggregate_connections

PATHS = ("prototype", "risk_fixed", "guard_risk_fixed", "prototype_risk_fixed")


def frozen_thresholds(energy, observed, labels, source, capacity):
    history = deque(source[-capacity:], maxlen=capacity)
    thresholds = []
    for i, value in enumerate(energy):
        threshold = conformal_threshold(history, 0.05)
        thresholds.append(np.nan if threshold is None else threshold)
        if observed[i] and labels[i] == 0:
            history.append(value)
    return np.array(thresholds)


def replay_path(
    values,
    reference,
    ratios,
    minutes,
    groups,
    keys,
    times,
    aggregation,
    thresholds,
    path,
    scope,
    checkpoint,
    *,
    continuous_filter=False,
    risk_key_values=None,
):
    if path not in PATHS:
        raise ValueError("Unknown prototype-temporal factorial path")
    if risk_key_values is not None and len(risk_key_values) != len(values):
        raise ValueError("Risk routing keys do not match frozen row energies")
    q95 = float(np.quantile(reference, 0.95))
    if q95 <= 0:
        raise ValueError("Prototype local score requires a positive source normal q95")
    sorted_reference = np.sort(reference)
    memory = CausalPeakMemory(q95) if path != "prototype" else None
    filtering = path.startswith("prototype")
    guarded = path != "risk_fixed"
    output, changed = np.zeros(len(minutes)), 0
    for i, minute in enumerate(minutes):
        ids = groups.get(int(minute), np.empty(0, dtype=np.int64))
        base = values[ids].copy()
        familiar = ratios[ids] <= 1 + 1e-6
        if filtering:
            local = np.isfinite(ratios[ids]) if continuous_filter else familiar
            base[local] = np.minimum(base[local], ratios[ids][local] ** 2 * q95)
        base_energy = (
            aggregate_connections(base, sorted_reference, aggregation, reference_sorted=True)
            if len(ids)
            else 0.0
        )
        predicted = base
        if memory is not None:
            active = (
                entity_keys({k: v[ids] for k, v in keys.items()}, scope)
                if risk_key_values is None
                else [risk_key_values[j] for j in ids]
            )
            shadow = None
            if i == len(minutes) // 2:
                torch.save(memory.state_dict(), checkpoint)
                shadow = CausalPeakMemory.from_state_dict(torch.load(checkpoint, weights_only=True))
            predicted = memory.predict((int(minute) + 1) * 60, active, base, times[ids])
            if shadow is not None:
                resumed = shadow.predict((int(minute) + 1) * 60, active, base, times[ids])
                if not np.array_equal(resumed, predicted):
                    raise ValueError("Prototype-risk disk resume changed actual prediction")
                memory = shadow
            if guarded:
                predicted[familiar] = base[familiar]
            contributors = (
                scan_contributors(base, sorted_reference, aggregation, reference_sorted=True)
                if len(ids) and base_energy > thresholds[i]
                else np.empty(0, dtype=np.int64)
            )
            if guarded:
                contributors = contributors[~familiar[contributors]]
            # No label reaches this temporal path. Normality affects only the
            # separately cached, strictly prior verified-normal prototypes.
            memory.feedback(verified_normal=False, contributors=contributors)
        output[i] = (
            aggregate_connections(predicted, sorted_reference, aggregation, reference_sorted=True)
            if len(ids)
            else 0.0
        )
        changed += int(not np.array_equal(predicted, values[ids]))
    return output, {
        "changed_windows": changed,
        "maximum_risk_keys": memory.maximum_seen if memory is not None else 0,
        "checkpoint_resumed_exact": memory is not None,
    }
