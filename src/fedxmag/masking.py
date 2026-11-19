"""Dependency-aware family corruption under graph-local scalar budgets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch
from torch import Tensor

from fedxmag.schema import DEFAULT_FEATURE_GROUPS, FeatureGroups, GraphBatch


@dataclass
class MaskedGraphBatch:
    """Encoder-visible values, primary targets, and dependency closures."""

    batch: GraphBatch
    node_visible: Tensor
    edge_visible: Tensor
    node_corrupted: Tensor
    edge_corrupted: Tensor
    node_targeted: Tensor
    edge_targeted: Tensor

    @property
    def node_targets(self) -> Tensor:
        return self.batch.node_features

    @property
    def edge_targets(self) -> Tensor:
        return self.batch.edge_features


@dataclass(frozen=True)
class _Candidate:
    kind: str
    entity: int
    family: str
    direction: str | None
    graph: int


class FamilyMasker:
    """Mask primary families and every deterministic node-edge copy of them."""

    def __init__(
        self,
        attribute_budget: float = 0.30,
        family_mask_probability: float = 0.35,
        minimum_visible_per_entity: int = 0,
        minimum_visible_per_graph: int = 1,
        groups: FeatureGroups = DEFAULT_FEATURE_GROUPS,
    ) -> None:
        if not 0.0 <= attribute_budget <= 1.0:
            raise ValueError("attribute_budget must be in [0, 1]")
        if not 0.0 <= family_mask_probability <= 1.0:
            raise ValueError("family_mask_probability must be in [0, 1]")
        if minimum_visible_per_entity < 0:
            raise ValueError("minimum_visible_per_entity must be nonnegative")
        if minimum_visible_per_graph < 0:
            raise ValueError("minimum_visible_per_graph must be nonnegative")
        self.attribute_budget = attribute_budget
        self.family_mask_probability = family_mask_probability
        self.minimum_visible_per_entity = minimum_visible_per_entity
        self.minimum_visible_per_graph = minimum_visible_per_graph
        self.groups = groups

    def _candidates(
        self,
        batch: GraphBatch,
        node_available: list[list[bool]],
        edge_available: list[list[bool]],
        node_graph_index: list[int],
        edge_graph_index: list[int],
    ) -> list[_Candidate]:
        result: list[_Candidate] = []
        for entity, availability in enumerate(node_available):
            for family in ("timing", "distribution"):
                for direction in ("out", "in"):
                    indices = self.groups.directional_node_indices(family, direction)
                    if any(availability[index] for index in indices):
                        result.append(
                            _Candidate(
                                "node",
                                entity,
                                family,
                                direction,
                                node_graph_index[entity],
                            )
                        )
        for entity, availability in enumerate(edge_available):
            for family in ("timing", "distribution"):
                indices = self.groups.indices("edge", family)
                if any(availability[index] for index in indices):
                    result.append(
                        _Candidate(
                            "edge",
                            entity,
                            family,
                            None,
                            edge_graph_index[entity],
                        )
                    )
        return result

    @staticmethod
    def _observed_flat_indices(
        availability: list[list[bool]],
        entity: int,
        indices: tuple[int, ...],
        width: int,
    ) -> set[int]:
        offset = entity * width
        return {offset + index for index in indices if availability[entity][index]}

    def _sparse_closure(
        self,
        candidate: _Candidate,
        node_available: list[list[bool]],
        edge_available: list[list[bool]],
        source: list[int],
        destination: list[int],
        outgoing: list[list[int]],
        incoming: list[list[int]],
    ) -> tuple[set[int], set[int], set[int], set[int]]:
        """Return sparse corruption and primary-target scalar indices."""

        node_width = self.groups.node_dim
        edge_width = self.groups.edge_dim
        node_closure: set[int] = set()
        edge_closure: set[int] = set()
        node_target: set[int] = set()
        edge_target: set[int] = set()

        if candidate.kind == "edge":
            edge_indices = self.groups.indices("edge", candidate.family)
            edge_target = self._observed_flat_indices(
                edge_available, candidate.entity, edge_indices, edge_width
            )
            edge_closure.update(edge_target)
            incident_edges = [candidate.entity]
        else:
            assert candidate.direction is not None
            node_indices = self.groups.directional_node_indices(
                candidate.family, candidate.direction
            )
            node_target = self._observed_flat_indices(
                node_available, candidate.entity, node_indices, node_width
            )
            node_closure.update(node_target)
            incident_edges = (
                outgoing[candidate.entity]
                if candidate.direction == "out"
                else incoming[candidate.entity]
            )

        edge_indices = self.groups.indices("edge", candidate.family)
        out_indices = self.groups.directional_node_indices(candidate.family, "out")
        in_indices = self.groups.directional_node_indices(candidate.family, "in")
        for edge in incident_edges:
            edge_closure.update(
                self._observed_flat_indices(
                    edge_available, edge, edge_indices, edge_width
                )
            )
            node_closure.update(
                self._observed_flat_indices(
                    node_available, source[edge], out_indices, node_width
                )
            )
            node_closure.update(
                self._observed_flat_indices(
                    node_available, destination[edge], in_indices, node_width
                )
            )
        return node_closure, edge_closure, node_target, edge_target

    def __call__(
        self,
        batch: GraphBatch,
        generator: torch.Generator | None = None,
    ) -> MaskedGraphBatch:
        node_available = batch.node_available.detach().cpu().tolist()
        edge_available = batch.edge_available.detach().cpu().tolist()
        node_graph_index = batch.node_graph_index.detach().cpu().tolist()
        edge_graph_index = batch.edge_graph_index.detach().cpu().tolist()
        source, destination = batch.edge_index.detach().cpu().tolist()
        outgoing: list[list[int]] = [[] for _ in node_available]
        incoming: list[list[int]] = [[] for _ in node_available]
        for edge, (source_node, destination_node) in enumerate(
            zip(source, destination, strict=True)
        ):
            outgoing[source_node].append(edge)
            incoming[destination_node].append(edge)

        candidates = self._candidates(
            batch,
            node_available,
            edge_available,
            node_graph_index,
            edge_graph_index,
        )
        remaining_node = [sum(row) for row in node_available]
        remaining_edge = [sum(row) for row in edge_available]
        graph_remaining = [0] * batch.num_graphs
        for graph, count in zip(node_graph_index, remaining_node, strict=True):
            graph_remaining[graph] += count
        for graph, count in zip(edge_graph_index, remaining_edge, strict=True):
            graph_remaining[graph] += count
        graph_budgets = [
            int(self.attribute_budget * available) for available in graph_remaining
        ]

        corrupted_node_indices: set[int] = set()
        corrupted_edge_indices: set[int] = set()
        targeted_node_indices: set[int] = set()
        targeted_edge_indices: set[int] = set()

        if candidates and any(graph_budgets) and self.family_mask_probability > 0.0:
            order = torch.randperm(len(candidates), generator=generator).tolist()
            graph_cost = [0] * batch.num_graphs
            for candidate_index in order:
                candidate = candidates[candidate_index]
                draw = float(torch.rand((), generator=generator))
                if draw > self.family_mask_probability:
                    continue
                add_node, add_edge, target_node, target_edge = self._sparse_closure(
                    candidate,
                    node_available,
                    edge_available,
                    source,
                    destination,
                    outgoing,
                    incoming,
                )
                if target_node & corrupted_node_indices or target_edge & corrupted_edge_indices:
                    continue
                new_node = add_node - corrupted_node_indices
                new_edge = add_edge - corrupted_edge_indices
                incremental_cost = len(new_node) + len(new_edge)
                if graph_cost[candidate.graph] + incremental_cost > graph_budgets[candidate.graph]:
                    continue
                node_decrements = Counter(index // self.groups.node_dim for index in new_node)
                edge_decrements = Counter(index // self.groups.edge_dim for index in new_edge)
                if any(
                    remaining_node[entity] - decrement < self.minimum_visible_per_entity
                    for entity, decrement in node_decrements.items()
                ) or any(
                    remaining_edge[entity] - decrement < self.minimum_visible_per_entity
                    for entity, decrement in edge_decrements.items()
                ):
                    continue
                if (
                    graph_remaining[candidate.graph] - incremental_cost
                    < self.minimum_visible_per_graph
                ):
                    continue
                corrupted_node_indices.update(new_node)
                corrupted_edge_indices.update(new_edge)
                targeted_node_indices.update(target_node)
                targeted_edge_indices.update(target_edge)
                for entity, decrement in node_decrements.items():
                    remaining_node[entity] -= decrement
                for entity, decrement in edge_decrements.items():
                    remaining_edge[entity] -= decrement
                graph_remaining[candidate.graph] -= incremental_cost
                graph_cost[candidate.graph] += incremental_cost

        node_corrupted = torch.zeros_like(batch.node_available)
        edge_corrupted = torch.zeros_like(batch.edge_available)
        node_targeted = torch.zeros_like(batch.node_available)
        edge_targeted = torch.zeros_like(batch.edge_available)
        for tensor, selected in (
            (node_corrupted, corrupted_node_indices),
            (edge_corrupted, corrupted_edge_indices),
            (node_targeted, targeted_node_indices),
            (edge_targeted, targeted_edge_indices),
        ):
            if selected:
                flat_indices = torch.tensor(
                    list(selected), dtype=torch.long, device=tensor.device
                )
                tensor.view(-1)[flat_indices] = True

        if not torch.all(node_corrupted <= batch.node_available):
            raise AssertionError("Node corruption selected a naturally unavailable value")
        if not torch.all(edge_corrupted <= batch.edge_available):
            raise AssertionError("Edge corruption selected a naturally unavailable value")
        if not torch.all(node_targeted <= node_corrupted):
            raise AssertionError("Node targets must be encoder-corrupted")
        if not torch.all(edge_targeted <= edge_corrupted):
            raise AssertionError("Edge targets must be encoder-corrupted")

        node_visible = batch.node_features.masked_fill(~batch.node_available, 0.0)
        edge_visible = batch.edge_features.masked_fill(~batch.edge_available, 0.0)
        node_visible = node_visible.masked_fill(node_corrupted, 0.0)
        edge_visible = edge_visible.masked_fill(edge_corrupted, 0.0)
        return MaskedGraphBatch(
            batch=batch,
            node_visible=node_visible,
            edge_visible=edge_visible,
            node_corrupted=node_corrupted,
            edge_corrupted=edge_corrupted,
            node_targeted=node_targeted,
            edge_targeted=edge_targeted,
        )
