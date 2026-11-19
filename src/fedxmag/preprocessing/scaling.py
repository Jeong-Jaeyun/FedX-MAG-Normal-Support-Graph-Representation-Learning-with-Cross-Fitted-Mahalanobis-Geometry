"""Availability-aware feature standardization fitted on normal training only."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from fedxmag.schema import DEFAULT_FEATURE_GROUPS, FeatureGroups, GraphSample


def _fit_kind(samples: Sequence[GraphSample], kind: str) -> tuple[Tensor, Tensor]:
    graph_means: list[Tensor] = []
    graph_observed: list[Tensor] = []
    for sample in samples:
        values = getattr(sample, f"{kind}_features")
        available = getattr(sample, f"{kind}_available")
        count = available.sum(dim=0)
        observed = count > 0
        mean = (values * available).sum(dim=0) / count.clamp_min(1)
        graph_means.append(mean)
        graph_observed.append(observed)

    means = torch.stack(graph_means)
    observed = torch.stack(graph_observed)
    graph_count = observed.sum(dim=0)
    if torch.any(graph_count == 0):
        missing = torch.nonzero(graph_count == 0).flatten().tolist()
        raise ValueError(f"Training data never observes {kind} feature indices {missing}")
    mean = (means * observed).sum(dim=0) / graph_count

    graph_variances: list[Tensor] = []
    for sample in samples:
        values = getattr(sample, f"{kind}_features")
        available = getattr(sample, f"{kind}_available")
        count = available.sum(dim=0)
        centered = (values - mean) * available
        graph_variances.append(centered.square().sum(dim=0) / count.clamp_min(1))
    per_graph_variance = torch.stack(graph_variances)
    variance = (per_graph_variance * observed).sum(dim=0) / graph_count
    scale = variance.sqrt().clamp_min(1e-6)
    return mean, scale


@dataclass(frozen=True)
class FeatureStandardizer:
    node_mean: Tensor
    node_scale: Tensor
    edge_mean: Tensor
    edge_scale: Tensor

    @classmethod
    def fit(cls, normal_training_samples: Sequence[GraphSample]) -> FeatureStandardizer:
        if not normal_training_samples:
            raise ValueError("Normal training samples are required")
        if any(sample.label != 0 for sample in normal_training_samples):
            raise ValueError("The standardizer may be fitted on normal training samples only")
        node_mean, node_scale = _fit_kind(normal_training_samples, "node")
        edge_mean, edge_scale = _fit_kind(normal_training_samples, "edge")
        return cls(node_mean, node_scale, edge_mean, edge_scale)

    def transform(
        self, sample: GraphSample, groups: FeatureGroups = DEFAULT_FEATURE_GROUPS
    ) -> GraphSample:
        node_features = (sample.node_features - self.node_mean) / self.node_scale
        edge_features = (sample.edge_features - self.edge_mean) / self.edge_scale
        node_features = node_features.masked_fill(~sample.node_available, 0.0)
        edge_features = edge_features.masked_fill(~sample.edge_available, 0.0)
        transformed = GraphSample(
            node_features=node_features,
            edge_features=edge_features,
            edge_index=sample.edge_index.clone(),
            node_available=sample.node_available.clone(),
            edge_available=sample.edge_available.clone(),
            label=sample.label,
            metadata=dict(sample.metadata),
        )
        transformed.validate(groups)
        return transformed

    def transform_many(
        self,
        samples: Sequence[GraphSample],
        groups: FeatureGroups = DEFAULT_FEATURE_GROUPS,
    ) -> list[GraphSample]:
        return [self.transform(sample, groups) for sample in samples]

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "node_mean": self.node_mean,
            "node_scale": self.node_scale,
            "edge_mean": self.edge_mean,
            "edge_scale": self.edge_scale,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Tensor]) -> FeatureStandardizer:
        required = {"node_mean", "node_scale", "edge_mean", "edge_scale"}
        if set(state) != required:
            raise ValueError(f"Expected scaler keys {sorted(required)}, got {sorted(state)}")
        return cls(**state)
