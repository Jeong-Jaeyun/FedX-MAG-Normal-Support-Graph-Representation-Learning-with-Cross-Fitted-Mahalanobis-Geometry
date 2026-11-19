"""Plan early verified-normal roles without allowing a model before feedback."""

from __future__ import annotations

import numpy as np


def minute_groups(observed):
    minutes = observed // 60
    order = np.argsort(minutes, kind="stable")
    unique, starts, counts = np.unique(minutes[order], return_index=True, return_counts=True)
    return {
        int(t): order[start : start + count]
        for t, start, count in zip(unique, starts, counts, strict=True)
    }


def early_normal_plans(data, window_minutes, labels, *, count=5, gap=300, delay=120):
    groups = minute_groups(data["observed_at"])
    offset = int(data.get("utc_offset_seconds", -10800))
    days = (window_minutes * 60 + offset) // 86400
    plans = []
    for day in np.unique(days):
        normal = [
            int(t)
            for t, y, d in zip(window_minutes, labels, days, strict=True)
            if y == 0 and d == day and int(t) in groups
        ]
        train_minutes = normal[:count]
        if len(train_minutes) < count:
            continue
        train = np.concatenate([groups[t] for t in train_minutes])
        train_end = int(data["observed_at"][train].max())
        selection_minutes = [
            t for t in normal if data["timestamp"][groups[t]].min() > train_end + gap
        ][:count]
        if len(selection_minutes) < count:
            continue
        selection = np.concatenate([groups[t] for t in selection_minutes])
        info_end = int(data["observed_at"][selection].max())
        plans.append(
            {
                "day": int(day),
                "train_minutes": train_minutes,
                "selection_minutes": selection_minutes,
                "train_rows": train,
                "selection_rows": selection,
                "reference_observation_end": train_end,
                "selection_observation_end": info_end,
                "feedback_available_at": (selection_minutes[-1] + 1) * 60,
                "ready_at": (selection_minutes[-1] + 1) * 60 + delay,
                "calibration_raw_start_after": info_end + gap,
            }
        )
    return plans


def normal_update_rows(
    data, source_roles, groups, window_minutes, labels, plan, *, maximum=20000, recent_maximum=5000
):
    if not 0 < recent_maximum <= maximum:
        raise ValueError("Recent normal reference budget must fit total reference budget")
    known = [
        int(t)
        for t, y in zip(window_minutes, labels, strict=True)
        if y == 0 and int(t) in groups and t <= plan["train_minutes"][-1]
    ]
    recent = np.concatenate([groups[t] for t in known])
    order = np.lexsort((data["signature"][recent], data["observed_at"][recent]))
    recent = recent[order[-recent_maximum:]]
    old = source_roles["train"]
    old = old[np.argsort(data["signature"][old])[: maximum - len(recent)]]
    reference = np.r_[old, recent]
    reference = reference[np.random.default_rng(17).permutation(len(reference))]
    selected = plan["selection_rows"]
    n = min(len(selected), 10000)
    selected = selected[np.argsort(data["signature"][selected])[:n]]
    # Keep the full original normal rank reference; reducing it to a tiny
    # balanced set would also change rank-scan's finite-sample resolution.
    selection = np.r_[source_roles["selection"], selected]
    if len(reference) < 32 or n < 32 or np.intersect1d(reference, selection).size:
        raise ValueError("Invalid disjoint normal commissioning rows")
    if np.any(data["label"][reference] != 0) or np.any(data["label"][selection] != 0):
        raise ValueError("An attack entered verified-normal commissioning")
    if data["observed_at"][recent].max() > plan["reference_observation_end"]:
        raise ValueError("Commissioning reference contains future feedback")
    return reference, selection


def latest_ready_stage(now, ready_times):
    # Same-second availability is conservative: use the old model at equality.
    eligible = np.flatnonzero(np.asarray(ready_times) < now)
    return int(eligible[-1]) + 1 if len(eligible) else 0
