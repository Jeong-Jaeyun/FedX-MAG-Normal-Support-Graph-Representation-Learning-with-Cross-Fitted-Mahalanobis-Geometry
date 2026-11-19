"""Tensor contracts for static Stage-1 graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import Tensor

EntityKind = Literal["node", "edge"]


@dataclass(frozen=True)
class FeatureGroups:
    """Ordered feature names and the family slices used by corruption and loss."""

    node_timing: tuple[str, ...]
    node_distribution: tuple[str, ...]
    edge_timing: tuple[str, ...]
    edge_distribution: tuple[str, ...]

    @property
    def node_names(self) -> tuple[str, ...]:
        return self.node_timing + self.node_distribution

    @property
    def edge_names(self) -> tuple[str, ...]:
        return self.edge_timing + self.edge_distribution

    @property
    def node_dim(self) -> int:
        return len(self.node_names)

    @property
    def edge_dim(self) -> int:
        return len(self.edge_names)

    def indices(self, kind: EntityKind, family: str) -> tuple[int, ...]:
        if family not in {"timing", "distribution"}:
            raise ValueError(f"Unknown feature family: {family}")
        timing = getattr(self, f"{kind}_timing")
        selected = getattr(self, f"{kind}_{family}")
        start = 0 if family == "timing" else len(timing)
        return tuple(range(start, start + len(selected)))

    def directional_node_indices(
        self, family: str, direction: Literal["out", "in"]
    ) -> tuple[int, ...]:
        """Return one directional half of a node feature family."""

        family_indices = self.indices("node", family)
        if direction not in {"out", "in"}:
            raise ValueError(f"Unknown node direction: {direction}")
        if len(family_indices) % 2:
            raise ValueError(f"Node {family} features must have equal out/in halves")
        midpoint = len(family_indices) // 2
        return family_indices[:midpoint] if direction == "out" else family_indices[midpoint:]


LEGACY_FEATURE_GROUPS = FeatureGroups(
    node_timing=(
        "log1p_out_record_iat_mean_ms",
        "log1p_out_record_iat_std_ms",
        "out_record_burstiness",
        "log1p_out_record_idle_p95_ms",
        "log1p_in_record_iat_mean_ms",
        "log1p_in_record_iat_std_ms",
        "in_record_burstiness",
        "log1p_in_record_idle_p95_ms",
    ),
    node_distribution=(
        "log1p_out_mean_packet_size_std_bytes",
        "log1p_out_mean_packet_size_iqr_bytes",
        "log1p_out_mean_packet_size_p90_bytes",
        "log1p_in_mean_packet_size_std_bytes",
        "log1p_in_mean_packet_size_iqr_bytes",
        "log1p_in_mean_packet_size_p90_bytes",
    ),
    edge_timing=(
        "log1p_record_iat_mean_ms",
        "log1p_record_iat_std_ms",
        "record_burstiness",
        "log1p_record_idle_p95_ms",
    ),
    edge_distribution=(
        "log1p_mean_packet_size_std_bytes",
        "log1p_mean_packet_size_iqr_bytes",
        "log1p_mean_packet_size_p90_bytes",
    ),
)


RESOLUTION_AWARE_V3A_FEATURE_GROUPS = FeatureGroups(
    node_timing=(
        "log1p_out_distinct_record_iat_median_s",
        "log1p_out_distinct_record_iat_iqr_s",
        "log1p_out_distinct_record_idle_p95_s",
        "out_same_timestamp_collision_fraction",
        "log1p_out_record_multiplicity_p95",
        "log1p_in_distinct_record_iat_median_s",
        "log1p_in_distinct_record_iat_iqr_s",
        "log1p_in_distinct_record_idle_p95_s",
        "in_same_timestamp_collision_fraction",
        "log1p_in_record_multiplicity_p95",
    ),
    node_distribution=LEGACY_FEATURE_GROUPS.node_distribution,
    edge_timing=(
        "log1p_distinct_record_iat_median_s",
        "log1p_distinct_record_iat_iqr_s",
        "log1p_distinct_record_idle_p95_s",
        "same_timestamp_collision_fraction",
        "log1p_record_multiplicity_p95",
    ),
    edge_distribution=LEGACY_FEATURE_GROUPS.edge_distribution,
)


RESOLUTION_AWARE_FEATURE_GROUPS = FeatureGroups(
    node_timing=(
        "log1p_out_distinct_record_iat_median_s",
        "log1p_out_distinct_record_iat_iqr_s",
        "log1p_out_distinct_record_idle_p95_s",
        "out_same_timestamp_collision_fraction",
        "log1p_in_distinct_record_iat_median_s",
        "log1p_in_distinct_record_iat_iqr_s",
        "log1p_in_distinct_record_idle_p95_s",
        "in_same_timestamp_collision_fraction",
    ),
    node_distribution=LEGACY_FEATURE_GROUPS.node_distribution,
    edge_timing=(
        "log1p_distinct_record_iat_median_s",
        "log1p_distinct_record_iat_iqr_s",
        "log1p_distinct_record_idle_p95_s",
        "same_timestamp_collision_fraction",
    ),
    edge_distribution=LEGACY_FEATURE_GROUPS.edge_distribution,
)


# Keep the historical default for direct library callers and archived v2 tests.
# Versioned entry points select their schema explicitly from configuration.
DEFAULT_FEATURE_GROUPS = LEGACY_FEATURE_GROUPS


def feature_groups_for_profile(profile: str) -> FeatureGroups:
    profiles = {
        "legacy_record_iat_v2": LEGACY_FEATURE_GROUPS,
        "distinct_second_multiplicity_v3": RESOLUTION_AWARE_V3A_FEATURE_GROUPS,
        "distinct_second_collision_v3": RESOLUTION_AWARE_FEATURE_GROUPS,
    }
    try:
        return profiles[profile]
    except KeyError as error:
        raise ValueError(f"Unknown timing feature profile: {profile}") from error


def _validate_entity_tensors(
    name: str,
    features: Tensor,
    available: Tensor,
    expected_dim: int,
) -> None:
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise ValueError(f"{name}_features must have shape [entities, {expected_dim}]")
    if available.shape != features.shape or available.dtype is not torch.bool:
        raise ValueError(f"{name}_available must be a bool tensor matching features")
    if not torch.isfinite(features).all():
        raise ValueError(f"{name}_features must be finite; encode missingness in availability")


@dataclass
class GraphSample:
    """One `L=1` graph before training-only artificial corruption."""

    node_features: Tensor
    edge_features: Tensor
    edge_index: Tensor
    node_available: Tensor
    edge_available: Tensor
    label: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, groups: FeatureGroups = DEFAULT_FEATURE_GROUPS) -> None:
        _validate_entity_tensors(
            "node", self.node_features, self.node_available, groups.node_dim
        )
        _validate_entity_tensors(
            "edge", self.edge_features, self.edge_available, groups.edge_dim
        )
        if self.node_features.shape[0] == 0:
            raise ValueError("A graph must contain at least one node")
        if self.edge_index.dtype is not torch.long or self.edge_index.ndim != 2:
            raise ValueError("edge_index must be a long tensor with shape [2, edges]")
        if self.edge_index.shape != (2, self.edge_features.shape[0]):
            raise ValueError("edge_index and edge_features disagree on the number of edges")
        if self.edge_index.numel():
            if int(self.edge_index.min()) < 0:
                raise ValueError("edge_index contains a negative node index")
            if int(self.edge_index.max()) >= self.node_features.shape[0]:
                raise ValueError("edge_index contains an out-of-range node index")
        if self.label not in {0, 1}:
            raise ValueError("label must be 0 (normal) or 1 (attack)")


@dataclass
class GraphBatch:
    """A disconnected union of graphs with explicit entity-to-graph maps."""

    node_features: Tensor
    edge_features: Tensor
    edge_index: Tensor
    node_available: Tensor
    edge_available: Tensor
    node_graph_index: Tensor
    edge_graph_index: Tensor
    labels: Tensor
    num_graphs: int

    def to(self, device: torch.device | str) -> GraphBatch:
        return GraphBatch(
            node_features=self.node_features.to(device),
            edge_features=self.edge_features.to(device),
            edge_index=self.edge_index.to(device),
            node_available=self.node_available.to(device),
            edge_available=self.edge_available.to(device),
            node_graph_index=self.node_graph_index.to(device),
            edge_graph_index=self.edge_graph_index.to(device),
            labels=self.labels.to(device),
            num_graphs=self.num_graphs,
        )


def collate_graphs(
    samples: list[GraphSample],
    groups: FeatureGroups = DEFAULT_FEATURE_GROUPS,
) -> GraphBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    node_features: list[Tensor] = []
    edge_features: list[Tensor] = []
    edge_indices: list[Tensor] = []
    node_available: list[Tensor] = []
    edge_available: list[Tensor] = []
    node_graph_index: list[Tensor] = []
    edge_graph_index: list[Tensor] = []
    labels: list[int] = []
    node_offset = 0

    for graph_index, sample in enumerate(samples):
        sample.validate(groups)
        num_nodes = sample.node_features.shape[0]
        num_edges = sample.edge_features.shape[0]
        node_features.append(sample.node_features)
        edge_features.append(sample.edge_features)
        edge_indices.append(sample.edge_index + node_offset)
        node_available.append(sample.node_available)
        edge_available.append(sample.edge_available)
        node_graph_index.append(
            torch.full((num_nodes,), graph_index, dtype=torch.long, device=sample.edge_index.device)
        )
        edge_graph_index.append(
            torch.full((num_edges,), graph_index, dtype=torch.long, device=sample.edge_index.device)
        )
        labels.append(sample.label)
        node_offset += num_nodes

    return GraphBatch(
        node_features=torch.cat(node_features, dim=0),
        edge_features=torch.cat(edge_features, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        node_available=torch.cat(node_available, dim=0),
        edge_available=torch.cat(edge_available, dim=0),
        node_graph_index=torch.cat(node_graph_index, dim=0),
        edge_graph_index=torch.cat(edge_graph_index, dim=0),
        labels=torch.tensor(labels, dtype=torch.long, device=samples[0].edge_index.device),
        num_graphs=len(samples),
    )
