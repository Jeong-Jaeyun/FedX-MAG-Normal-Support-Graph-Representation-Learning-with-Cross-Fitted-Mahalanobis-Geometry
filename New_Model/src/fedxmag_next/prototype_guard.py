"""Past verified-normal equality-key prototypes in fixed source geometry."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .daily_detector import DailyDeepEnergyDetector
from .deep_neighbors import whiten
from .experiments import knn_scores

SCOPES = ("source", "peer")


def entity_keys(keys, scope):
    if scope not in SCOPES:
        raise ValueError("Unknown prototype routing scope")
    source = np.asarray(keys["source"])
    destination = np.asarray(keys["destination"])
    if (
        source.dtype != np.uint64
        or destination.dtype != np.uint64
        or source.shape != destination.shape
    ):
        raise ValueError("Prototype keys must be aligned uint64 equality metadata")
    return (
        source.tolist()
        if scope == "source"
        else list(zip(source.tolist(), destination.tolist(), strict=True))
    )


@torch.no_grad()
def keyed_distances(reference, query, reference_keys, query_keys, scope):
    rkeys, qkeys = entity_keys(reference_keys, scope), entity_keys(query_keys, scope)
    if any(len(v) != len(rkeys) for v in reference.values()) or any(
        len(v) != len(qkeys) for v in query.values()
    ):
        raise ValueError("Prototype features and equality keys differ")
    references, queries = defaultdict(list), defaultdict(list)
    for i, key in enumerate(rkeys):
        references[key].append(i)
    for i, key in enumerate(qkeys):
        queries[key].append(i)
    result = {name: torch.full((len(qkeys),), torch.inf, dtype=torch.float64) for name in query}
    for key, ids in queries.items():
        if key not in references:
            continue
        old = references[key]
        for name, values in query.items():
            result[name][ids] = knn_scores(reference[name][old], values[ids], k=1)
    return result


class PrototypeProjector:
    def __init__(self, state):
        self.state = state
        self.base = DailyDeepEnergyDetector(state)
        if self.base.encoder_kind != "dependency":
            raise ValueError("Prototype guard requires the fixed dependency encoder")
        for model in self.base.models:
            model.double().eval()

    @torch.no_grad()
    def project(self, features):
        if features.ndim != 2 or features.shape[1] != len(self.base.feature_names):
            raise ValueError("Prototype projection schema differs")
        if not torch.isfinite(features).all():
            raise ValueError("Prototype projection requires finite observations")
        values = self.base.normalize_features(features).double()
        raw = whiten(values, self.state["raw_supports"]["raw_mahalanobis_m1"])
        deep = []
        for model, item in zip(self.base.models, self.state["seed_models"], strict=True):
            embedded = (
                torch.cat([model.layers[:-1](chunk) for chunk in values.split(8192)])
                if len(values)
                else values.new_empty((0, len(item["embedding_mean"])))
            )
            embedded = (embedded - item["embedding_mean"].double()) / item[
                "embedding_scale"
            ].double()
            deep.append(whiten(embedded, item["supports"]["1"]["geometry"]))
        return {"raw": raw, "deep": torch.cat(deep, dim=1)}


def fit_radii(reference, selection, reference_keys, selection_keys):
    radii, counts = {}, {}
    for scope in SCOPES:
        distances = keyed_distances(reference, selection, reference_keys, selection_keys, scope)
        radii[scope], counts[scope] = {}, {}
        for name, values in distances.items():
            finite = values[torch.isfinite(values)]
            if len(finite) < 32:
                raise ValueError("Too few source selection normals with same-key reference")
            radii[scope][name] = max(float(torch.quantile(finite, 0.95)), 1e-6)
            counts[scope][name] = len(finite)
    return radii, counts


class VerifiedPrototypeMemory:
    def __init__(self, values, keys, times, radii, maximum=5000):
        if maximum <= 0 or not len(times):
            raise ValueError("A prototype memory requires positive capacity and source normals")
        if any(len(v) != len(times) for v in values.values()) or any(
            len(v) != len(times) for v in keys.values()
        ):
            raise ValueError("Initial normal prototype metadata differ")
        order = np.argsort(times, kind="stable")[-maximum:]
        self.values = {k: v[order].double().clone() for k, v in values.items()}
        self.keys = {k: np.asarray(v)[order].copy() for k, v in keys.items()}
        self.times = np.asarray(times)[order].astype(np.int64)
        self.radii, self.maximum = radii, maximum
        self.last_time, self.pending = int(self.times.max()), None
        self.maximum_seen = len(order)

    def query(self, now, values, keys, times):
        times = np.asarray(times, dtype=np.int64)
        if self.pending is not None or now <= self.last_time:
            raise ValueError("Commit prototype feedback before a strictly later query")
        if np.any(times < self.last_time) or np.any(times >= now):
            raise ValueError("Prototype query times leave the current completed window")
        ratios = {}
        for scope in SCOPES:
            distances = keyed_distances(self.values, values, self.keys, keys, scope)
            ratios[scope] = {k: v / self.radii[scope][k] for k, v in distances.items()}
        self.pending = (
            now,
            {k: v.double().clone() for k, v in values.items()},
            {k: np.asarray(v).copy() for k, v in keys.items()},
            times.copy(),
        )
        return ratios

    def feedback(self, *, verified_normal):
        if self.pending is None:
            raise ValueError("Prototype admission must follow prediction")
        now, values, keys, times = self.pending
        if verified_normal and len(times):
            order = np.argsort(times, kind="stable")
            self.values = {
                k: torch.cat((v, values[k][order]))[-self.maximum :] for k, v in self.values.items()
            }
            self.keys = {
                k: np.concatenate((v, keys[k][order]))[-self.maximum :]
                for k, v in self.keys.items()
            }
            self.times = np.concatenate((self.times, times[order]))[-self.maximum :]
        self.last_time, self.pending = now, None
        self.maximum_seen = max(self.maximum_seen, len(self.times))

    def state_dict(self):
        if self.pending is not None:
            raise ValueError("Save prototype memory only after committed feedback")
        return {
            "format_version": 1,
            "values": self.values,
            "keys": {k: torch.from_numpy(v.view(np.int64).copy()) for k, v in self.keys.items()},
            "times": torch.from_numpy(self.times.copy()),
            "radii": self.radii,
            "maximum": self.maximum,
            "last_time": self.last_time,
            "maximum_seen": self.maximum_seen,
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("format_version") != 1:
            raise ValueError("Unsupported verified-prototype checkpoint")
        memory = cls(
            state["values"],
            {k: v.numpy().view(np.uint64) for k, v in state["keys"].items()},
            state["times"].numpy(),
            state["radii"],
            state["maximum"],
        )
        memory.last_time, memory.maximum_seen = state["last_time"], state["maximum_seen"]
        return memory
