"""Frozen-DNN local normal support in Euclidean and source-Mahalanobis metrics."""

from __future__ import annotations

import copy

import torch

from .daily_detector import DailyDeepEnergyDetector
from .experiments import knn_scores


def whiten(values, geometry):
    if geometry["means"].shape[0] != 1:
        raise ValueError("Neighbor whitening requires one frozen global normal metric")
    return torch.linalg.solve_triangular(
        geometry["cholesky"][0], (values.double() - geometry["means"][0]).T, upper=False
    ).T.contiguous()


def normalized_neighbor_score(reference, values, center, scale):
    return (knn_scores(reference, values, k=20) - center) / scale + 1


class DeepNeighborDetector:
    def __init__(self, state):
        if state.get("format_version") != 1:
            raise ValueError("Unsupported frozen deep-neighbor detector")
        self.state = state
        self.base = DailyDeepEnergyDetector(state["base_encoder"])
        # Frozen learned coefficients are unchanged; double forward arithmetic
        # avoids batch-dependent float32 noise near duplicate normal neighbors.
        self.models = [model.double().eval() for model in self.base.models]

    @classmethod
    def load(cls, path):
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    @torch.no_grad()
    def score(self, features):
        if features.ndim != 2 or features.shape[1] != len(self.base.feature_names):
            raise ValueError("Neighbor query feature schema differs")
        if not torch.isfinite(features).all():
            raise ValueError("Neighbor queries must be finite")
        values = self.base.normalize_features(features)
        raw = self.state["raw_whitened"]
        scored = {
            "raw_whitened_knn": normalized_neighbor_score(
                raw["reference"], whiten(values, raw["geometry"]), raw["center"], raw["scale"]
            )
        }
        deep = {"euclidean": [], "whitened": []}
        for model, item, bank in zip(
            self.models, self.base.state["seed_models"], self.state["deep_banks"], strict=True
        ):
            embedded = model.layers[:-1](values.double())
            embedded = (embedded - item["embedding_mean"].double()) / item[
                "embedding_scale"
            ].double()
            for metric in deep:
                entry = bank[metric]
                query = whiten(embedded, entry["geometry"]) if metric == "whitened" else embedded
                deep[metric].append(
                    normalized_neighbor_score(
                        entry["reference"], query, entry["center"], entry["scale"]
                    )
                )
        for metric, energies in deep.items():
            scored[f"deep_{metric}_knn_ensemble"] = torch.stack(energies).mean(0)
        return scored


@torch.no_grad()
def fit_neighbor_state(base_state, selection_features):
    if base_state.get("encoder_kind", "dependency") != "dependency":
        raise ValueError("This ablation freezes the original dependency encoder")
    base = DailyDeepEnergyDetector(base_state)
    reference = base_state["raw_reference"]
    selected = base.normalize_features(selection_features)
    state = {
        "format_version": 1,
        "base_encoder": copy.deepcopy(base_state),
        "neighbors": 20,
        "reference_count": len(reference),
        "forward_dtype": "float64",
        "deep_banks": [],
    }

    def bank(ref, query, geometry=None):
        if geometry is not None:
            ref, query = whiten(ref, geometry), whiten(query, geometry)
        distance = knn_scores(ref, query, k=20)
        center, q95 = torch.quantile(distance, torch.tensor([0.5, 0.95]).double())
        entry = {"reference": ref, "center": float(center), "scale": max(float(q95 - center), 1e-6)}
        if geometry is not None:
            entry["geometry"] = geometry
        return entry

    state["raw_whitened"] = bank(
        reference, selected, base_state["raw_supports"]["raw_mahalanobis_m1"]
    )
    for model, item in zip(base.models, base_state["seed_models"], strict=True):
        model.double().eval()
        ref, query = (model.layers[:-1](v.double()) for v in (reference, selected))
        ref = (ref - item["embedding_mean"].double()) / item["embedding_scale"].double()
        query = (query - item["embedding_mean"].double()) / item["embedding_scale"].double()
        state["deep_banks"].append(
            {
                "euclidean": bank(ref, query),
                "whitened": bank(ref, query, item["supports"]["1"]["geometry"]),
            }
        )
        print("fit frozen neighbor references seed", item["seed"], flush=True)
    return state
