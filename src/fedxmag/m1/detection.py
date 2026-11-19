"""Attack-independent Mahalanobis detection, baselines, and evaluation metrics.

Thresholds come only from calibration-normal scores. Matrix algebra uses torch;
numpy is used only for sorting-based statistics.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fedxmag.m1.covariance import (
    CovarianceEstimate,
    estimate_covariance,
    mahalanobis_energy,
    spectral_diagnostics,
)
from fedxmag.schema import FeatureGroups, GraphSample

STRUCTURAL_DESCRIPTOR_NAMES = (
    "log1p_nodes",
    "log1p_edges",
    "density",
    "log1p_records",
    "log1p_timing_observed_records",
    "reverse_observation_fraction",
    "log1p_records_per_edge",
    "mean_out_degree",
    "std_out_degree",
    "max_out_degree_share",
    "std_in_degree",
    "max_in_degree_share",
    "reciprocity",
    "source_only_node_fraction",
    "sink_only_node_fraction",
    "weak_component_fraction",
    "largest_weak_component_fraction",
)


def _weak_components(nodes: int, source: list[int], destination: list[int]) -> list[int]:
    parent = list(range(nodes))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left, right in zip(source, destination, strict=True):
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right
    sizes: dict[int, int] = {}
    for item in range(nodes):
        root = find(item)
        sizes[root] = sizes.get(root, 0) + 1
    return sorted(sizes.values(), reverse=True)


def structural_descriptors(sample: GraphSample) -> list[float]:
    """Handcrafted structure and volume descriptors (MAG-ED-style family).

    Uses only the directed edge set and record counts. Protocol, labels, attack
    metadata, and endpoint identities are never read.
    """

    nodes = int(sample.node_features.shape[0])
    edges = int(sample.edge_features.shape[0])
    source, destination = (row.tolist() for row in sample.edge_index)
    out_degree = np.bincount(source, minlength=nodes).astype(np.float64)
    in_degree = np.bincount(destination, minlength=nodes).astype(np.float64)
    pairs = set(zip(source, destination, strict=True))
    reciprocal = sum((right, left) in pairs for left, right in pairs)
    components = _weak_components(nodes, source, destination)
    records = float(sample.metadata.get("event_count", 0))
    timing_records = float(sample.metadata.get("timing_observed_event_count", 0))
    reverse_records = float(sample.metadata.get("synthetic_reverse_event_count", 0))
    possible = nodes * (nodes - 1)
    return [
        math.log1p(nodes),
        math.log1p(edges),
        edges / possible if possible else 0.0,
        math.log1p(records),
        math.log1p(timing_records),
        reverse_records / records if records else 0.0,
        math.log1p(records / edges) if edges else 0.0,
        float(out_degree.mean()),
        float(out_degree.std()),
        float(out_degree.max() / edges) if edges else 0.0,
        float(in_degree.std()),
        float(in_degree.max() / edges) if edges else 0.0,
        reciprocal / edges if edges else 0.0,
        float(np.mean((out_degree > 0) & (in_degree == 0))),
        float(np.mean((in_degree > 0) & (out_degree == 0))),
        len(components) / nodes,
        components[0] / nodes,
    ]


def attribute_statistics(sample: GraphSample, groups: FeatureGroups) -> list[float]:
    """Pooled mean, standard deviation, and availability of every behavioural feature."""

    values: list[float] = []
    for kind in ("node", "edge"):
        features = getattr(sample, f"{kind}_features").to(torch.float64)
        available = getattr(sample, f"{kind}_available")
        for column in range(getattr(groups, f"{kind}_dim")):
            observed = features[available[:, column], column]
            count = int(observed.numel())
            values.append(float(observed.mean()) if count else 0.0)
            values.append(float(observed.std(unbiased=False)) if count > 1 else 0.0)
            values.append(count / max(int(features.shape[0]), 1))
    return values


def descriptor_matrix(
    samples: Sequence[GraphSample], kind: str, groups: FeatureGroups
) -> Tensor:
    if kind == "structural":
        rows = [structural_descriptors(sample) for sample in samples]
    elif kind == "attribute":
        rows = [attribute_statistics(sample, groups) for sample in samples]
    elif kind == "combined":
        rows = [
            structural_descriptors(sample) + attribute_statistics(sample, groups)
            for sample in samples
        ]
    else:
        raise ValueError(f"Unknown descriptor kind: {kind}")
    return torch.tensor(rows, dtype=torch.float64)


@dataclass
class Standardizer:
    """Train-normal z-scoring that drops columns constant on train-normal.

    A constant reference column carries no normal-support information and would
    otherwise give an unshrunk covariance an exactly singular direction.
    """

    mean: Tensor
    scale: Tensor
    keep: Tensor

    @classmethod
    def fit(cls, values: Tensor) -> Standardizer:
        scale = values.std(dim=0, unbiased=True)
        keep = scale > 1e-12
        return cls(values.mean(dim=0), torch.where(keep, scale, torch.ones_like(scale)), keep)

    def __call__(self, values: Tensor) -> Tensor:
        return ((values - self.mean) / self.scale)[:, self.keep]


@dataclass
class MahalanobisDetector:
    estimate: CovarianceEstimate
    standardizer: Standardizer | None = None

    @classmethod
    def fit(
        cls,
        reference: Tensor,
        estimator: str = "ledoit_wolf",
        *,
        standardize: bool = False,
        fixed_alpha: float = 0.5,
        alpha_override: float | None = None,
    ) -> MahalanobisDetector:
        standardizer = Standardizer.fit(reference) if standardize else None
        values = standardizer(reference) if standardizer is not None else reference
        if alpha_override is not None:
            estimate = estimate_covariance(values, "fixed", fixed_alpha=alpha_override)
        else:
            estimate = estimate_covariance(values, estimator, fixed_alpha=fixed_alpha)
        return cls(estimate, standardizer)

    @property
    def dimension(self) -> int:
        return int(self.estimate.mean.shape[0])

    def score(self, values: Tensor) -> Tensor:
        if self.standardizer is not None:
            values = self.standardizer(values)
        return mahalanobis_energy(values, self.estimate)

    def diagnostics(self) -> dict[str, float]:
        return {
            "dimension": float(self.dimension),
            "alpha": float(self.estimate.alpha),
            "scale": float(self.estimate.scale),
            **{f"shape_{key}": value for key, value in spectral_diagnostics(
                self.estimate.shape
            ).items()},
            **{
                f"raw_{key}": value
                for key, value in spectral_diagnostics(
                    self.estimate.sample_covariance / self.estimate.scale
                ).items()
            },
        }


def conformal_threshold(calibration_scores: Sequence[float], quantile: float) -> float:
    """``ceil((n + 1) q)``-th smallest calibration score (split-conformal)."""

    ordered = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    rank = math.ceil((ordered.size + 1) * quantile)
    return float(ordered[min(rank, ordered.size) - 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """One-based ranks with ties sharing their average rank."""

    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    last = np.cumsum(counts).astype(np.float64)
    return (last - (counts - 1) / 2.0)[inverse]


def auroc(normal: Sequence[float], attack: Sequence[float]) -> float:
    """Mann-Whitney AUROC with ties counted as one half (attack is positive)."""

    normal_array = np.asarray(normal, dtype=np.float64)
    attack_array = np.asarray(attack, dtype=np.float64)
    ranks = _average_ranks(np.concatenate((normal_array, attack_array)))
    positive_rank_sum = ranks[normal_array.size :].sum()
    positives = attack_array.size
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * normal_array.size)
    )


def average_precision(normal: Sequence[float], attack: Sequence[float]) -> float:
    """Step-wise average precision over distinct score thresholds."""

    scores = np.concatenate((np.asarray(normal, float), np.asarray(attack, float)))
    labels = np.concatenate((np.zeros(len(normal)), np.ones(len(attack))))
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    distinct = np.flatnonzero(np.diff(scores)) if scores.size > 1 else np.array([], int)
    cut = np.concatenate((distinct, [scores.size - 1]))
    true_positive = np.cumsum(labels)[cut]
    flagged = cut + 1.0
    precision = true_positive / flagged
    recall = true_positive / labels.sum()
    previous = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous) * precision))


def threshold_metrics(
    normal: Sequence[float], attack: Sequence[float], threshold: float
) -> dict[str, float]:
    normal_array = np.asarray(normal, dtype=np.float64)
    attack_array = np.asarray(attack, dtype=np.float64)
    true_positive = float(np.sum(attack_array > threshold))
    false_positive = float(np.sum(normal_array > threshold))
    false_negative = attack_array.size - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive else 0.0
    recall = true_positive / attack_array.size
    return {
        "threshold": threshold,
        "tpr": recall,
        "fpr": false_positive / normal_array.size,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_negatives": false_negative,
    }


def detection_summary(
    calibration: Sequence[float],
    normal: Sequence[float],
    attack: Sequence[float],
    fpr_targets: Sequence[float],
) -> dict[str, Any]:
    """All threshold-based metrics use calibration-normal thresholds only."""

    return {
        "auroc": auroc(normal, attack),
        "auprc": average_precision(normal, attack),
        "attack_prevalence": len(attack) / (len(attack) + len(normal)),
        "at_calibrated_fpr": {
            f"{target:g}": threshold_metrics(
                normal, attack, conformal_threshold(calibration, 1.0 - target)
            )
            for target in fpr_targets
        },
    }


def cluster_bootstrap_detection(
    scores: Mapping[str, Mapping[str, np.ndarray]],
    members: Mapping[str, Sequence[str]],
    clusters: Mapping[str, Sequence[str]],
    pairs: Sequence[tuple[str, str]],
    fpr_targets: Sequence[float],
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired capture-date cluster bootstrap of AUROC and calibrated TPR.

    ``scores[key]`` holds ``calibration``, ``normal``, and ``attack`` arrays for
    one fitted detector (e.g. one model seed); ``members[method]`` lists the keys
    whose metrics are averaged within each replicate. ``clusters[role]`` gives
    the capture date of every graph in that role. Each replicate resamples whole
    dates once and reuses the same draw for every method, so differences are
    paired, and calibration thresholds are re-derived from the resampled
    calibration windows so threshold uncertainty is included.
    """

    roles = ("calibration", "normal", "attack")
    all_clusters = sorted({item for role in roles for item in clusters[role]})
    position = {cluster: index for index, cluster in enumerate(all_clusters)}
    indices = {role: np.asarray([position[item] for item in clusters[role]]) for role in roles}
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(
        len(all_clusters), np.full(len(all_clusters), 1.0 / len(all_clusters)), size=resamples
    )
    metrics = ("auroc", *(f"tpr@{target:g}" for target in fpr_targets))
    per_key: dict[str, dict[str, list[float]]] = {
        key: {metric: [] for metric in metrics} for key in scores
    }
    for draw in draws:
        expanded = {
            role: np.repeat(np.arange(index.size), draw[index]) for role, index in indices.items()
        }
        if not all(expanded[role].size for role in roles):
            continue
        for key, arrays in scores.items():
            normal = arrays["normal"][expanded["normal"]]
            attack = arrays["attack"][expanded["attack"]]
            calibration = arrays["calibration"][expanded["calibration"]]
            per_key[key]["auroc"].append(auroc(normal, attack))
            for target in fpr_targets:
                threshold = conformal_threshold(calibration, 1.0 - target)
                per_key[key][f"tpr@{target:g}"].append(float(np.mean(attack > threshold)))
    replicates = {
        method: {
            metric: np.mean(
                [np.asarray(per_key[key][metric]) for key in keys], axis=0
            )
            for metric in metrics
        }
        for method, keys in members.items()
    }
    result: dict[str, Any] = {
        "resamples_used": int(len(next(iter(per_key.values()))["auroc"])),
        "clusters": len(all_clusters),
        "intervals": {
            method: {
                metric: {
                    "lower": float(np.quantile(values, 0.025)),
                    "upper": float(np.quantile(values, 0.975)),
                }
                for metric, values in method_values.items()
            }
            for method, method_values in replicates.items()
        },
        "differences": {},
    }
    for left, right in pairs:
        entry: dict[str, Any] = {}
        for metric in metrics:
            difference = replicates[left][metric] - replicates[right][metric]
            entry[metric] = {
                "mean_difference": float(difference.mean()),
                "lower": float(np.quantile(difference, 0.025)),
                "upper": float(np.quantile(difference, 0.975)),
                "probability_not_positive": float(np.mean(difference <= 0.0)),
            }
        result["differences"][f"{left}__minus__{right}"] = entry
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Rank correlation using elementwise operations only (no BLAS)."""

    x = _average_ranks(np.asarray(left, dtype=np.float64))
    y = _average_ranks(np.asarray(right, dtype=np.float64))
    x -= x.mean()
    y -= y.mean()
    denominator = math.sqrt(float(np.sum(x * x)) * float(np.sum(y * y)))
    return float(np.sum(x * y)) / denominator if denominator else float("nan")
