"""Chronological, observation-disjoint whole-window inputs for H3."""

import json
from pathlib import Path

import numpy as np
import torch

from .experiments import sha256
from .h1_experiment import PACKETS
from .window_bridge import census, census_sources

HISTORY = 8
CONTEXT_MINUTES = 5
ROLES = ("train", "selection", "residual_support", "calibration")


def chronological_roles(normal_minutes, history=HISTORY, context_minutes=CONTEXT_MINUTES):
    """Allocate complete sequences chronologically, then own parent windows.

    Sparse normal minutes can leave parent-count quantiles with no usable
    sequence. Allocate the largest feasible 55/15/15/15 sequence quota instead,
    retaining at least32 examples per role and purging all overlapping slots.
    Only source-normal chronology determines the boundaries.
    """
    normal = np.asarray(normal_minutes, dtype=np.int64)
    if len(normal) == 0 or np.any(np.diff(normal) <= 0):
        raise ValueError("Sorted unique normal minutes required")
    span = history + context_minutes
    normal_set = set(normal.tolist())
    valid = np.array(
        [m for m in normal if all(int(m) - j in normal_set for j in range(span + 1))],
        dtype=np.int64,
    )
    cuts = None
    for total in range(len(valid), 127, -1):
        # Distribute rounding remainder to calibration, before checking minima.
        quotas = [int(0.55 * total), int(0.15 * total), int(0.15 * total)]
        quotas.append(total - sum(quotas))
        if min(quotas) < 32:
            continue
        start, ends = 0, []
        for quota in quotas:
            end = start + quota - 1
            if end >= len(valid):
                break
            ends.append(valid[end])
            start = int(np.searchsorted(valid, valid[end] + span, side="right"))
        if len(ends) == len(ROLES):
            cuts = ends[:-1]
            break
    if cuts is None:
        raise ValueError("Insufficient complete source-normal sequences for four disjoint roles")
    boundaries = [
        0,
        *[int(np.searchsorted(normal, cut, side="right")) for cut in cuts],
        len(normal),
    ]
    parents, examples = {}, {}
    for role, a, b in zip(ROLES, boundaries[:-1], boundaries[1:], strict=True):
        owned = normal[a:b]
        owned_set = set(owned.tolist())
        parents[role] = owned
        examples[role] = np.array(
            [
                m
                for m in owned
                if all(int(m) - j in owned_set for j in range(history + context_minutes + 1))
            ],
            dtype=np.int64,
        )
    return parents, examples


def histories(minutes, z_minutes, z, available, history=HISTORY):
    """Exact elapsed-minute slots, oldest first; missing slots are zero plus mask."""
    if len(z_minutes) != len(z) or len(available) != len(z) or np.any(np.diff(z_minutes) <= 0):
        raise ValueError("Aligned sorted unique timeline required")
    lookup = {int(m): i for i, m in enumerate(z_minutes)}
    x = np.zeros((len(minutes), history, z.shape[1]), dtype=np.float64)
    mask = np.zeros((len(minutes), history), dtype=np.float64)
    for i, m in enumerate(minutes):
        for j in range(history):
            p = lookup.get(int(m) - history + j)
            if p is not None and available[p]:
                x[i, j], mask[i, j] = z[p], 1
    return x, mask


def summarize(features, timestamps):
    """Identical label-free operator in each domain: mean, SD, log count, presence."""
    row_minutes = timestamps // 60
    order = np.argsort(row_minutes, kind="stable")
    ordered = row_minutes[order]
    starts = np.flatnonzero(np.r_[True, np.diff(ordered) != 0])
    minutes = ordered[starts]
    summary = np.empty((len(starts), features.shape[1] * 2 + 2), dtype=np.float64)
    for k, (a, b) in enumerate(zip(starts, np.r_[starts[1:], len(order)], strict=True)):
        values = features[order[a:b]].astype(np.float64)
        summary[k] = np.r_[values.mean(0), values.std(0), np.log1p(b - a), 1.0]
    if not np.isfinite(summary).all():
        raise ValueError("Nonfinite whole-window summary")
    return minutes, summary


def load_inputs(root, dataset):
    root = Path(root).resolve()
    packet_path = root / "New_Model/candidates/prototype_structure_v3" / f"{PACKETS[dataset]}.pt"
    packet = torch.load(packet_path, weights_only=True)
    risk = root / packet["prototype_model"]["protocol"]["args"]["run"]
    reg = json.loads((risk / "registration.json").read_text(encoding="utf-8"))
    data_path = root / reg["args"]["data"]
    digest = sha256(data_path)
    if digest != packet["data_sha256"]:
        raise ValueError("Prepared input changed from frozen packet")
    target = torch.load(risk / "scores.pt", weights_only=True)
    target_minutes = target["minutes"].numpy()
    labels = target["labels"].numpy()
    source_paths = census_sources(root, dataset)
    source_provenance = [
        {
            "path": str(p),
            "bytes": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
            "sha256": sha256(p),
        }
        for p in source_paths
    ]
    truth = census(source_paths)
    for path, saved in zip(source_paths, source_provenance, strict=True):
        if path.stat().st_size != saved["bytes"] or path.stat().st_mtime_ns != saved["mtime_ns"]:
            raise ValueError("Full-table census input changed during source audit")
    if any(truth[int(m)]["label"] != int(y) for m, y in zip(target_minutes, labels, strict=True)):
        raise ValueError("Full-table target labels changed from frozen census")
    with np.load(data_path) as data:
        ts, row_labels = data["observed_at"], data["label"]
        observed_minutes, summary = summarize(data["features"], ts)
        feature_names = data["feature_names"].tolist()
    source_normal = np.array(
        [
            m
            for m in observed_minutes
            if m < target_minutes.min() and int(m) in truth and truth[int(m)]["label"] == 0
        ],
        dtype=np.int64,
    )
    parents, examples = chronological_roles(source_normal)
    row_ids = {name: np.flatnonzero(np.isin(ts // 60, owned)) for name, owned in parents.items()}
    for ids in row_ids.values():
        if not np.all(row_labels[ids] == 0) or np.any(ts[ids] // 60 >= target_minutes.min()):
            raise ValueError("Source role not wholly normal and prior")
    for i, name in enumerate(ROLES):
        for other in ROLES[i + 1 :]:
            if np.intersect1d(row_ids[name], row_ids[other]).size:
                raise ValueError("Source observation IDs overlap")
            if np.intersect1d(parents[name], parents[other]).size:
                raise ValueError("Source parent window IDs overlap")
        minimum = 32 if name == "residual_support" else 20 if name == "calibration" else 2
        if len(examples[name]) < minimum:
            raise ValueError(
                f"Insufficient disjoint complete H3 sequences: {name}={len(examples[name])}"
            )
    all_minutes = np.union1d(observed_minutes, target_minutes)
    all_summary = np.zeros((len(all_minutes), summary.shape[1]))
    available = np.zeros(len(all_minutes), dtype=bool)
    positions = np.searchsorted(all_minutes, observed_minutes)
    all_summary[positions], available[positions] = summary, True
    audit = {
        "dataset": dataset,
        "data": str(data_path),
        "data_sha256": digest,
        "full_table_census_sources": source_provenance,
        "parent_window_counts": {k: len(v) for k, v in parents.items()},
        "example_counts": {k: len(v) for k, v in examples.items()},
        "observation_counts": {k: len(v) for k, v in row_ids.items()},
        "source_whole_normal_windows": len(source_normal),
        "target_census_windows": len(target_minutes),
        "target_missing_current": int(
            (~available[np.searchsorted(all_minutes, target_minutes)]).sum()
        ),
        "feature_names": feature_names,
        "role_parent_window_intersections": 0,
        "role_observation_intersections": 0,
        "history_minutes": HISTORY,
        "feature_context_minutes": CONTEXT_MINUTES,
        "cutoff_minute": int(target_minutes.min()),
        "minimum_boundary_example_separation_minutes": min(
            int(examples[b].min() - examples[a].max())
            for a, b in zip(ROLES[:-1], ROLES[1:], strict=True)
        ),
    }
    past_attack = np.array(
        [
            any(truth.get(int(m) - lag, {}).get("label", 0) == 1 for lag in range(1, HISTORY + 1))
            for m in target_minutes
        ],
        dtype=bool,
    )
    return dict(
        minutes=all_minutes,
        summary=all_summary,
        available=available,
        parents=parents,
        examples=examples,
        row_ids=row_ids,
        target_minutes=target_minutes,
        labels=labels,
        types=target["types"],
        past_attack=past_attack,
        audit=audit,
    )
