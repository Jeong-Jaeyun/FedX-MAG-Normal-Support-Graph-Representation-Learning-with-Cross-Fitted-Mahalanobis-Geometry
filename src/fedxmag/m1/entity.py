"""Entity-level normal support: node and edge state energies with graph aggregation.

Graph-level mean pooling averages a few attack-involved entities with many
normal background entities. Here each node (or edge) state is scored against a
train-normal entity reference, and a graph's score aggregates its most
anomalous entities. Matrix algebra uses torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fedxmag.confirmation import build_masked_batches, per_graph_family_errors
from fedxmag.losses import m0_loss
from fedxmag.m1.covariance import (
    ESTIMATORS,
    differentiable_shape_terms,
    estimate_covariance,
    mahalanobis_energy,
    spectral_diagnostics,
)
from fedxmag.m1.detection import MahalanobisDetector, auroc
from fedxmag.m1.training import clean_view
from fedxmag.masking import FamilyMasker, MaskedGraphBatch
from fedxmag.models import AdjacencyShuffledGraphModel, M0GraphModel, M0Output, SetContextPredictor
from fedxmag.models.m0 import _encoder_values, _shuffle_endpoints_within_graph
from fedxmag.schema import FeatureGroups, GraphSample, collate_graphs

AGGREGATIONS = ("max", "top3_mean", "mean")


class StateGraphModel(M0GraphModel):
    """M0GraphModel that also returns final node and edge states.

    Parameters and state-dict keys are identical, so every M0/M1 graph checkpoint
    loads unchanged.
    """

    def forward_with_states(self, masked: MaskedGraphBatch) -> tuple[M0Output, Tensor, Tensor]:
        batch = masked.batch
        node_state = self.node_input(_encoder_values(masked, "node"))
        edge_state = self.edge_input(_encoder_values(masked, "edge"))
        for layer in self.layers:
            node_state, edge_state = layer(node_state, edge_state, batch.edge_index)
        return super().forward(masked), node_state, edge_state


@dataclass
class EntityStates:
    """Per-graph entity states for one role."""

    node: list[Tensor]
    edge: list[Tensor]

    def stacked(self, kind: str) -> Tensor:
        parts = [item for item in getattr(self, kind) if item.shape[0]]
        return torch.cat(parts, dim=0)


@torch.no_grad()
def encode_entities(
    model: StateGraphModel,
    samples: Sequence[GraphSample],
    groups: FeatureGroups,
    batch_size: int = 32,
) -> EntityStates:
    model.eval()
    nodes: list[Tensor] = []
    edges: list[Tensor] = []
    for start in range(0, len(samples), batch_size):
        batch = collate_graphs(list(samples[start : start + batch_size]), groups)
        _, node_state, edge_state = model.forward_with_states(clean_view(batch))
        for graph in range(batch.num_graphs):
            nodes.append(node_state[batch.node_graph_index == graph].to(torch.float64))
            edges.append(edge_state[batch.edge_graph_index == graph].to(torch.float64))
    return EntityStates(nodes, edges)


@torch.no_grad()
def encode_control_entities(
    model: SetContextPredictor | AdjacencyShuffledGraphModel,
    samples: Sequence[GraphSample],
    groups: FeatureGroups,
    batch_size: int = 32,
) -> EntityStates:
    """Final entity states of the equal-capacity Set-context or Shuffled-graph control."""

    model.eval()
    shuffled = isinstance(model, AdjacencyShuffledGraphModel)
    nodes: list[Tensor] = []
    edges: list[Tensor] = []
    for start in range(0, len(samples), batch_size):
        batch = collate_graphs(list(samples[start : start + batch_size]), groups)
        masked = clean_view(batch)
        if shuffled:
            masked = _shuffle_endpoints_within_graph(
                masked, model.shuffle_seed, model.endpoint_shuffle_mode
            )
        batch = masked.batch
        node_state = model.node_input(_encoder_values(masked, "node"))
        edge_state = model.edge_input(_encoder_values(masked, "edge"))
        for layer in model.layers:
            if shuffled:
                node_state, edge_state = layer(node_state, edge_state, batch.edge_index)
            else:
                node_state, edge_state = layer(
                    node_state, edge_state, batch.node_graph_index, batch.edge_graph_index,
                    batch.num_graphs,
                )
        for graph in range(batch.num_graphs):
            nodes.append(node_state[batch.node_graph_index == graph].to(torch.float64))
            edges.append(edge_state[batch.edge_graph_index == graph].to(torch.float64))
    return EntityStates(nodes, edges)


def stratified_aurocs(
    normal: np.ndarray,
    attack: np.ndarray,
    normal_nodes: np.ndarray,
    attack_nodes: np.ndarray,
    normal_dates: np.ndarray,
    attack_dates: np.ndarray,
    attack_fraction: np.ndarray,
) -> dict[str, float | None]:
    """AUROC within node-count quartiles, within shared capture dates, and by dilution."""

    combined = np.concatenate((normal_nodes, attack_nodes))
    edges = list(np.unique(np.quantile(combined, np.linspace(0, 1, 5)))[:-1]) + [np.inf]

    def weighted(groups: list[tuple[np.ndarray, np.ndarray]], minimum: int) -> float | None:
        weights, values = [], []
        for normal_mask, attack_mask in groups:
            if normal_mask.sum() >= minimum and attack_mask.sum() >= minimum:
                weights.append(float(normal_mask.sum() * attack_mask.sum()))
                values.append(auroc(normal[normal_mask], attack[attack_mask]))
        return float(np.average(values, weights=weights)) if values else None

    size = weighted(
        [
            ((normal_nodes >= low) & (normal_nodes < high),
             (attack_nodes >= low) & (attack_nodes < high))
            for low, high in zip(edges[:-1], edges[1:], strict=True)
        ],
        5,
    )
    shared = sorted(set(normal_dates.tolist()) & set(attack_dates.tolist()))
    date = weighted([(normal_dates == d, attack_dates == d) for d in shared], 3)
    result: dict[str, float | None] = {
        "size_stratified_auroc": size,
        "within_date_auroc": date,
        "shared_dates": float(len(shared)),
    }
    for label, low, high in (("fraction<0.1", 0.0, 0.1), ("0.1<=fraction<0.5", 0.1, 0.5),
                             ("fraction>=0.5", 0.5, 1.01)):
        mask = (attack_fraction >= low) & (attack_fraction < high)
        result[f"auroc[{label}]"] = auroc(normal, attack[mask]) if mask.sum() >= 5 else None
    return result


def localization_rate(
    entity_scores: Sequence[Tensor],
    endpoint_keys: Sequence[Sequence[str]],
    attack_endpoints: Sequence[set[str]],
) -> dict[str, float]:
    """How often the highest-energy node of an attack window is an attack endpoint."""

    hits, chance = [], []
    for scores, keys, endpoints in zip(entity_scores, endpoint_keys, attack_endpoints, strict=True):
        flags = np.asarray([key in endpoints for key in keys])
        if scores.numel() == 0 or not flags.any() or flags.all():
            continue
        hits.append(bool(flags[int(torch.argmax(scores))]))
        chance.append(float(flags.mean()))
    return {
        "windows": float(len(hits)),
        "top1_is_attack_endpoint": float(np.mean(hits)) if hits else float("nan"),
        "random_node_is_attack_endpoint": float(np.mean(chance)) if chance else float("nan"),
    }


def raw_entity_features(samples: Sequence[GraphSample]) -> EntityStates:
    """Standardized features with unavailable values zeroed, plus availability flags."""

    def rows(sample: GraphSample, kind: str) -> Tensor:
        values = getattr(sample, f"{kind}_features").to(torch.float64)
        available = getattr(sample, f"{kind}_available").to(torch.float64)
        return torch.cat((values * available, available), dim=1)

    return EntityStates(
        [rows(sample, "node") for sample in samples],
        [rows(sample, "edge") for sample in samples],
    )


def aggregate(scores: Sequence[Tensor], how: str) -> np.ndarray:
    if how not in AGGREGATIONS:
        raise ValueError(f"Unknown aggregation: {how}")
    output = []
    for values in scores:
        if values.numel() == 0:
            output.append(0.0)
        elif how == "max":
            output.append(float(values.max()))
        elif how == "top3_mean":
            output.append(float(values.sort().values[-3:].mean()))
        else:
            output.append(float(values.mean()))
    return np.asarray(output, dtype=np.float64)


@dataclass
class EntityScorer:
    """Ledoit-Wolf Mahalanobis over train-normal entities of one kind."""

    detector: MahalanobisDetector
    kind: str

    @classmethod
    def fit(
        cls, reference: EntityStates, kind: str, estimator: str = "ledoit_wolf"
    ) -> EntityScorer:
        return cls(MahalanobisDetector.fit(reference.stacked(kind), estimator), kind)

    def entity_scores(self, states: EntityStates) -> list[Tensor]:
        return [
            self.detector.score(item) if item.shape[0] else item.new_zeros(0)
            for item in getattr(states, self.kind)
        ]

    def graph_scores(self, states: EntityStates, how: str) -> np.ndarray:
        return aggregate(self.entity_scores(states), how)


@dataclass(frozen=True)
class EntitySupportSettings:
    """Entity-level M1: cross-fitted energy of node/edge states.

    The rank guardrail acts on the raw reference shape ``S_R / s_R`` (the
    formulation's shrunk-shape guardrail never binds under heavy shrinkage). No
    condition-number term is used: LayerNorm states have one exact null
    direction, so their raw condition number is always infinite.
    """

    estimator: str = "ledoit_wolf"
    kinds: tuple[str, ...] = ("node", "edge")
    lambda_m: float = 1.0
    lambda_c: float = 1.0
    lambda_s: float = 1.0
    rank_min: float = 12.0
    scale_min: float = 0.01
    scale_max: float = 100.0
    reconstruction_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.estimator not in ESTIMATORS:
            raise ValueError(f"Unknown estimator: {self.estimator}")
        if not set(self.kinds) <= {"node", "edge"} or not self.kinds:
            raise ValueError("kinds must be a nonempty subset of {'node', 'edge'}")


def _graph_balanced(values: Tensor, graph_index: Tensor, graphs: Tensor) -> Tensor:
    """Mean over graphs of the mean entity value within each graph."""

    positions = torch.searchsorted(graphs, graph_index)
    sums = values.new_zeros(graphs.numel()).index_add_(0, positions, values)
    counts = values.new_zeros(graphs.numel()).index_add_(
        0, positions, torch.ones_like(values)
    )
    present = counts > 0
    return (sums[present] / counts[present]).mean()


def entity_support_loss(
    states: Mapping[str, Tensor],
    graph_index: Mapping[str, Tensor],
    num_graphs: int,
    settings: EntitySupportSettings,
    generator: torch.Generator,
) -> tuple[Tensor, dict[str, float]]:
    """Symmetric cross-fitted entity energy with graph-disjoint reference/query halves."""

    order = torch.randperm(num_graphs, generator=generator)
    half = num_graphs // 2
    halves = (order[:half].sort().values, order[half : 2 * half].sort().values)
    energies: list[Tensor] = []
    ranks: list[Tensor] = []
    scales: list[Tensor] = []
    diagnostics: dict[str, float] = {}
    for kind in settings.kinds:
        values, index = states[kind], graph_index[kind]
        dimension = values.shape[1]
        members = [torch.isin(index, part) for part in halves]
        for query_part, reference_rows, query_rows in (
            (halves[1], members[0], members[1]),
            (halves[0], members[1], members[0]),
        ):
            if int(reference_rows.sum()) < 2 or int(query_rows.sum()) < 1:
                continue
            estimate = estimate_covariance(values[reference_rows], settings.estimator)
            energy = mahalanobis_energy(values[query_rows], estimate) / dimension
            energies.append(_graph_balanced(energy, index[query_rows], query_part))
            raw_shape = estimate.sample_covariance / estimate.scale
            _, effective_rank = differentiable_shape_terms(raw_shape)
            ranks.append(torch.relu(settings.rank_min - effective_rank).square())
            scales.append(
                torch.relu(settings.scale_min - estimate.scale).square()
                + torch.relu(estimate.scale - settings.scale_max).square()
            )
            diagnostics[f"{kind}_alpha"] = float(estimate.alpha)
            diagnostics[f"{kind}_raw_effective_rank"] = float(effective_rank.detach())
    if not energies:
        zero = next(iter(states.values())).sum() * 0.0
        return zero, diagnostics
    energy = torch.stack(energies).mean()
    rank = torch.stack(ranks).mean()
    scale = torch.stack(scales).mean()
    diagnostics.update(
        {
            "energy_over_d": float(energy.detach()),
            "rank_penalty": float(rank.detach()),
            "scale_penalty": float(scale.detach()),
        }
    )
    total = settings.lambda_m * energy + settings.lambda_c * rank + settings.lambda_s * scale
    return total, diagnostics


def train_entity_m1(
    model: StateGraphModel,
    train_samples: Sequence[GraphSample],
    select_samples: Sequence[GraphSample],
    masker: FamilyMasker,
    settings: EntitySupportSettings,
    *,
    epochs: int,
    batch_size: int,
    validation_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    early_stopping_patience: int,
    validation_mask_seeds: Sequence[int],
    loss_weights: Mapping[str, float],
    seed: int,
    groups: FeatureGroups,
) -> list[dict[str, Any]]:
    """Joint reconstruction + entity-level normal support; selects on reconstruction."""

    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    split_generator = torch.Generator().manual_seed(seed + 7_919)
    cached = [
        build_masked_batches(select_samples, masker, mask_seed, validation_batch_size, groups)
        for mask_seed in validation_mask_seeds
    ]
    best, stale = float("inf"), 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    full_batches = len(train_samples) // batch_size
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_samples), generator=generator).tolist()
        sums: dict[str, float] = {}
        for index in range(full_batches):
            chosen = order[index * batch_size : (index + 1) * batch_size]
            batch = collate_graphs([train_samples[item] for item in chosen], groups)
            masked = masker(batch, generator=generator)
            output, node_state, edge_state = model.forward_with_states(masked)
            reconstruction = m0_loss(output, masked, weights=loss_weights, groups=groups)
            support, diagnostics = entity_support_loss(
                {"node": node_state, "edge": edge_state},
                {"node": batch.node_graph_index, "edge": batch.edge_graph_index},
                batch.num_graphs,
                settings,
                split_generator,
            )
            total = settings.reconstruction_weight * reconstruction.total + support.to(
                reconstruction.total.dtype
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            record = {"reconstruction": float(reconstruction.total.detach()),
                      "total": float(total.detach()), **diagnostics}
            for name, value in record.items():
                sums[name] = sums.get(name, 0.0) + value
        model.eval()
        with torch.no_grad():
            family = np.mean(
                [np.nanmean(per_graph_family_errors(model, batches, groups), axis=0)
                 for batches in cached],
                axis=0,
            )
            states = encode_entities(model, select_samples, groups)
            geometry = {
                kind: spectral_diagnostics(torch.cov(states.stacked(kind).T))
                for kind in ("node", "edge")
            }
        metric = float(np.max(family))
        history.append(
            {
                "epoch": epoch,
                "training": {name: value / full_batches for name, value in sums.items()},
                "validation": {
                    "family_mse": family.tolist(),
                    "worst_family_mse": metric,
                    "clean_select_node_effective_rank": geometry["node"]["effective_rank_entropy"],
                    "clean_select_edge_effective_rank": geometry["edge"]["effective_rank_entropy"],
                },
                "selection_metric_value": metric,
            }
        )
        if metric < best:
            best, stale = metric, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= early_stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history
