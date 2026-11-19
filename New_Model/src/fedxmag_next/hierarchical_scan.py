"""Nested rank evidence within active endpoints, then across a whole window.

Keys only group current observations. Normal group references forget identity;
this does not fit a separate model to a persistent host or train a GNN.
"""

from __future__ import annotations

import numpy as np
import torch

from .daily_detector import DailyDeepEnergyDetector, support_energy
from .geometry import ConditionalMahalanobis

SCOPES = ("source", "destination", "pair")


def grouping(keys, scope):
    if scope not in SCOPES:
        raise ValueError("Unknown grouping scope")
    source, destination = np.asarray(keys["source"]), np.asarray(keys["destination"])
    if source.ndim != 1 or destination.shape != source.shape:
        raise ValueError("Endpoint grouping arrays must align")
    values = np.stack((source, destination), axis=1) if scope == "pair" else keys[scope]
    _, inverse, counts = np.unique(values, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    return np.split(order, np.cumsum(counts)[:-1]) if len(source) else []


def surprises(values, sorted_normal):
    values, sorted_normal = np.asarray(values), np.asarray(sorted_normal)
    if len(sorted_normal) == 0 or not np.isfinite(values).all():
        raise ValueError("Normal rank evidence requires finite observations and a reference")
    # Prespecified numerical tie tolerance, fitted on no target data.
    tolerance = 1e-6 * np.abs(values) + 1e-9
    p = (1 + len(sorted_normal) - np.searchsorted(sorted_normal, values - tolerance)) / (
        len(sorted_normal) + 1
    )
    return -np.log(p)


def scan_surprise(values):
    if not len(values):
        raise ValueError("A rank scan needs at least one observation")
    return float(
        np.max(np.log(np.arange(1, len(values) + 1) / len(values)) + np.sort(values)[::-1])
    )


def group_statistics(row_scores, keys, scope, row_reference):
    groups = grouping(keys, scope)
    if any(len(v) != len(keys["source"]) for v in row_scores.values()):
        raise ValueError("Row energies do not align with endpoint groups")
    output = {}
    for name, values in row_scores.items():
        ranked = surprises(values, row_reference[name])
        output[name] = np.array([scan_surprise(ranked[g]) for g in groups])
    return output


@torch.no_grad()
def edge_scores(detector, features):
    """Exactly the existing raw/deep energies, omitting unused expensive kNN."""
    if features.ndim != 2 or features.shape[1] != len(detector.feature_names):
        raise ValueError("Hierarchical scan input schema differs")
    if not torch.isfinite(features).all():
        raise ValueError("Hierarchical scan inputs must be finite")
    normalized = detector.normalize_features(features)
    hidden = []
    for model, item in zip(detector.models, detector.state["seed_models"], strict=True):
        embedded = model.layers[:-1](normalized)
        hidden.append((embedded - item["embedding_mean"]) / item["embedding_scale"])
    result = {}
    for modes in (1, 8):
        raw = ConditionalMahalanobis.from_state_dict(
            detector.state["raw_supports"][f"raw_mahalanobis_m{modes}"]
        )
        result[f"raw_mahalanobis_m{modes}"] = support_energy(raw, normalized).numpy()
        deep = []
        for embedded, item in zip(hidden, detector.state["seed_models"], strict=True):
            support = item["supports"][str(modes)]
            energy = support_energy(
                ConditionalMahalanobis.from_state_dict(support["geometry"]), embedded
            )
            deep.append((energy - support["score_center"]) / support["score_scale"] + 1)
        result[f"deep_mahalanobis_m{modes}_ensemble"] = torch.stack(deep).mean(0).numpy()
    return result


class HierarchicalScanDetector:
    def __init__(self, state):
        if state.get("format_version") != 1:
            raise ValueError("Unsupported hierarchical scan artifact")
        self.state = state
        self.base = DailyDeepEnergyDetector(state["base_encoder"])
        self.row_reference = {k: v.numpy() for k, v in state["row_reference"].items()}

    @classmethod
    def load(cls, path):
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    def aggregate(self, row_scores, keys):
        output = {}
        for scope in SCOPES:
            grouped = group_statistics(row_scores, keys, scope, self.row_reference)
            for name, values in grouped.items():
                reference = self.state["group_reference"][scope][name].numpy()
                output[f"{name}_nested_{scope}"] = scan_surprise(surprises(values, reference))
        return output

    def score_window(self, features, keys):
        return self.aggregate(edge_scores(self.base, features), keys)
