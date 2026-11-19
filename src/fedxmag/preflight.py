"""Dataset diagnostics emitted before any model training is allowed."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fedxmag.config import config_fingerprint
from fedxmag.masking import FamilyMasker
from fedxmag.preprocessing.events import IngestionReport
from fedxmag.schema import DEFAULT_FEATURE_GROUPS, FeatureGroups, GraphSample, collate_graphs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {key: float("nan") for key in ("min", "p25", "median", "p75", "p95", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def _availability(
    samples: Sequence[GraphSample], kind: str, groups: FeatureGroups = DEFAULT_FEATURE_GROUPS
) -> dict[str, float]:
    names = getattr(groups, f"{kind}_names")
    rates: list[torch.Tensor] = []
    for sample in samples:
        available = getattr(sample, f"{kind}_available")
        rates.append(available.to(torch.float32).mean(dim=0))
    if not rates:
        return {name: float("nan") for name in names}
    graph_macro = torch.stack(rates).mean(dim=0)
    return {name: float(graph_macro[index]) for index, name in enumerate(names)}


def _shortcut_rates(
    samples: Sequence[GraphSample], groups: FeatureGroups = DEFAULT_FEATURE_GROUPS
) -> dict[str, dict[str, float | int]]:
    totals: Counter[str] = Counter()
    matches: Counter[str] = Counter()
    for sample in samples:
        source, destination = sample.edge_index
        for family in ("timing", "distribution"):
            edge_indices = groups.indices("edge", family)
            for relation, node_rows, direction in (
                ("source_out", source, "out"),
                ("destination_in", destination, "in"),
            ):
                node_indices = groups.directional_node_indices(family, direction)
                node_values = sample.node_features[node_rows][:, list(node_indices)]
                node_available = sample.node_available[node_rows][:, list(node_indices)]
                edge_values = sample.edge_features[:, list(edge_indices)]
                edge_available = sample.edge_available[:, list(edge_indices)]
                jointly_available = node_available & edge_available
                key = f"{family}_{relation}"
                totals[key] += int(jointly_available.sum())
                matches[key] += int(
                    (torch.isclose(node_values, edge_values, atol=1e-7, rtol=1e-6)
                    & jointly_available).sum()
                )
    return {
        key: {
            "jointly_available_scalars": totals[key],
            "exact_match_scalars": matches[key],
            "exact_match_rate": matches[key] / totals[key] if totals[key] else float("nan"),
        }
        for key in sorted(totals)
    }


def _reciprocal_timing_exact_match(
    samples: Sequence[GraphSample],
    groups: FeatureGroups = DEFAULT_FEATURE_GROUPS,
) -> dict[str, Any]:
    """Measure identical timing scalars across reciprocal directed edges once per pair."""

    timing_indices = list(groups.indices("edge", "timing"))
    reciprocal_pairs = 0
    pairs_with_joint_timing = 0
    pairs_with_all_timing = 0
    pairs_with_all_timing_exact = 0
    jointly_available_scalars = 0
    exact_match_scalars = 0
    per_feature_available: Counter[str] = Counter()
    per_feature_matches: Counter[str] = Counter()
    for sample in samples:
        source, destination = sample.edge_index.tolist()
        edge_by_nodes = {
            (int(src), int(dst)): edge
            for edge, (src, dst) in enumerate(zip(source, destination, strict=True))
        }
        for (src, dst), edge in edge_by_nodes.items():
            if src >= dst:
                continue
            reverse = edge_by_nodes.get((dst, src))
            if reverse is None:
                continue
            reciprocal_pairs += 1
            available = (
                sample.edge_available[edge, timing_indices]
                & sample.edge_available[reverse, timing_indices]
            )
            matches = torch.isclose(
                sample.edge_features[edge, timing_indices],
                sample.edge_features[reverse, timing_indices],
                atol=1e-7,
                rtol=1e-6,
            )
            count = int(available.sum())
            if count:
                pairs_with_joint_timing += 1
                jointly_available_scalars += count
                exact_match_scalars += int((matches & available).sum())
            if bool(available.all()):
                pairs_with_all_timing += 1
                if bool(matches.all()):
                    pairs_with_all_timing_exact += 1
            for offset, name in enumerate(groups.edge_timing):
                if bool(available[offset]):
                    per_feature_available[name] += 1
                    if bool(matches[offset]):
                        per_feature_matches[name] += 1
    rate = (
        exact_match_scalars / jointly_available_scalars
        if jointly_available_scalars
        else 0.0
    )
    possible_scalars = reciprocal_pairs * len(timing_indices)
    return {
        "reciprocal_edge_pairs": reciprocal_pairs,
        "pairs_with_jointly_available_timing": pairs_with_joint_timing,
        "pairs_with_all_timing_features_jointly_available": pairs_with_all_timing,
        "pairs_with_all_timing_features_exact": pairs_with_all_timing_exact,
        "jointly_available_timing_scalars": jointly_available_scalars,
        "exact_match_timing_scalars": exact_match_scalars,
        "reciprocal_exact_match_rate": rate,
        "reciprocal_exact_match_rate_definition": (
            "exact matches divided by jointly available reciprocal timing scalars"
        ),
        "reciprocal_exact_match_exposure_rate": (
            exact_match_scalars / possible_scalars if possible_scalars else 0.0
        ),
        "reciprocal_joint_timing_pair_rate": (
            pairs_with_joint_timing / reciprocal_pairs if reciprocal_pairs else 0.0
        ),
        "full_vector_exact_match_rate": (
            pairs_with_all_timing_exact / pairs_with_all_timing
            if pairs_with_all_timing
            else 0.0
        ),
        "per_feature": {
            name: {
                "jointly_available_pairs": per_feature_available[name],
                "exact_match_pairs": per_feature_matches[name],
                "exact_match_rate": (
                    per_feature_matches[name] / per_feature_available[name]
                    if per_feature_available[name]
                    else 0.0
                ),
            }
            for name in groups.edge_timing
        },
    }


def _mask_diagnostics(
    samples: Sequence[GraphSample],
    masker: FamilyMasker,
    mask_seeds: Sequence[int],
    groups: FeatureGroups = DEFAULT_FEATURE_GROUPS,
    batch_size: int = 32,
) -> dict[str, Any]:
    targeted = Counter()
    available = Counter()
    corrupted = 0
    primary = 0
    graph_runs = 0
    graphs_with_targets = 0
    degree_targeted = Counter()
    degree_targeted_graphs = Counter()
    degree_available = Counter()
    availability_targeted = Counter()
    availability_available = Counter()

    def degree_bin(value: int) -> str:
        if value <= 1:
            return "0-1"
        if value <= 3:
            return "2-3"
        return "4+"

    for seed in mask_seeds:
        generator = torch.Generator().manual_seed(seed)
        for start in range(0, len(samples), batch_size):
            batch = collate_graphs(list(samples[start : start + batch_size]), groups)
            masked = masker(batch, generator=generator)
            source, destination = batch.edge_index
            out_degree = torch.bincount(source, minlength=batch.node_features.shape[0])
            in_degree = torch.bincount(destination, minlength=batch.node_features.shape[0])
            batch_has_target = torch.zeros(batch.num_graphs, dtype=torch.bool)
            batch_degree_graph_targets: set[tuple[str, int]] = set()
            for kind, target, availability, graph_index in (
                ("node", masked.node_targeted, batch.node_available, batch.node_graph_index),
                ("edge", masked.edge_targeted, batch.edge_available, batch.edge_graph_index),
            ):
                for family in ("timing", "distribution"):
                    key = f"{kind}_{family}"
                    indices = list(groups.indices(kind, family))
                    targeted[key] += int(target[:, indices].sum())
                    available[key] += int(availability[:, indices].sum())
                targeted_rows = target.any(dim=1)
                if bool(targeted_rows.any()):
                    batch_has_target[graph_index[targeted_rows]] = True

            for family in ("timing", "distribution"):
                for direction, degrees in (("out", out_degree), ("in", in_degree)):
                    indices = list(groups.directional_node_indices(family, direction))
                    for entity in range(batch.node_features.shape[0]):
                        observed = int(batch.node_available[entity, indices].sum())
                        if not observed:
                            continue
                        dependency_bin = degree_bin(int(degrees[entity]))
                        degree_key = (
                            f"node_{family}_{direction}|degree={dependency_bin}"
                        )
                        availability_key = (
                            f"node_{family}_{direction}|availability="
                            f"{'full' if observed == len(indices) else 'partial'}"
                        )
                        selected = int(masked.node_targeted[entity, indices].sum())
                        degree_available[degree_key] += observed
                        degree_targeted[degree_key] += selected
                        if selected:
                            batch_degree_graph_targets.add(
                                (degree_key, int(batch.node_graph_index[entity]))
                            )
                        availability_available[availability_key] += observed
                        availability_targeted[availability_key] += selected

                edge_indices = list(groups.indices("edge", family))
                for edge in range(batch.edge_features.shape[0]):
                    observed = int(batch.edge_available[edge, edge_indices].sum())
                    if not observed:
                        continue
                    dependency_degree = max(
                        int(out_degree[source[edge]]), int(in_degree[destination[edge]])
                    )
                    degree_key = f"edge_{family}|degree={degree_bin(dependency_degree)}"
                    availability_key = (
                        f"edge_{family}|availability="
                        f"{'full' if observed == len(edge_indices) else 'partial'}"
                    )
                    selected = int(masked.edge_targeted[edge, edge_indices].sum())
                    degree_available[degree_key] += observed
                    degree_targeted[degree_key] += selected
                    if selected:
                        batch_degree_graph_targets.add(
                            (degree_key, int(batch.edge_graph_index[edge]))
                        )
                    availability_available[availability_key] += observed
                    availability_targeted[availability_key] += selected
            graphs_with_targets += int(batch_has_target.sum())
            for degree_key, _ in batch_degree_graph_targets:
                degree_targeted_graphs[degree_key] += 1
            graph_runs += batch.num_graphs
            corrupted += int(masked.node_corrupted.sum() + masked.edge_corrupted.sum())
            primary += int(masked.node_targeted.sum() + masked.edge_targeted.sum())
    return {
        "primary_scalar_rates": {
            key: targeted[key] / available[key] if available[key] else float("nan")
            for key in sorted(available)
        },
        "dependency_expansion_ratio": corrupted / primary if primary else float("nan"),
        "graph_target_rate": graphs_with_targets / graph_runs if graph_runs else float("nan"),
        "primary_scalar_rates_by_dependency_degree": {
            key: degree_targeted[key] / degree_available[key]
            for key in sorted(degree_available)
        },
        "available_scalars_by_dependency_degree": dict(sorted(degree_available.items())),
        "targeted_scalars_by_dependency_degree": dict(sorted(degree_targeted.items())),
        "targeted_graphs_by_dependency_degree": dict(
            sorted(degree_targeted_graphs.items())
        ),
        "primary_scalar_rates_by_availability": {
            key: availability_targeted[key] / availability_available[key]
            for key in sorted(availability_available)
        },
        "mask_seeds": list(mask_seeds),
    }


def _protocol_distribution(samples: Sequence[GraphSample]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for sample in samples:
        counts.update(sample.metadata.get("protocol_counts", {}))
    total = sum(counts.values())
    return {key: value / total for key, value in sorted(counts.items())} if total else {}


def _total_variation(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def build_preflight_report(
    graph_splits: Mapping[str, Sequence[GraphSample]],
    ingestion: IngestionReport,
    input_paths: Sequence[Path],
    config: dict[str, Any],
    masker: FamilyMasker,
    groups: FeatureGroups = DEFAULT_FEATURE_GROUPS,
) -> dict[str, Any]:
    all_samples = [sample for samples in graph_splits.values() for sample in samples]
    train_protocols = _protocol_distribution(graph_splits["train"])
    split_stats: dict[str, Any] = {}
    identities: dict[str, set[tuple[str, int]]] = {}
    sessions: dict[str, set[str]] = {}
    for name, samples in graph_splits.items():
        protocols = _protocol_distribution(samples)
        session_set = {str(sample.metadata["session_key"]) for sample in samples}
        identities[name] = {
            (str(sample.metadata["session_key"]), int(sample.metadata["window_start_ns"]))
            for sample in samples
        }
        sessions[name] = session_set
        categories = Counter(
            category
            for sample in samples
            for category in sample.metadata.get("attack_categories", [])
        )
        subcategories = Counter(
            subcategory
            for sample in samples
            for subcategory in sample.metadata.get("attack_subcategories", [])
        )
        split_stats[name] = {
            "graphs": len(samples),
            "normal_graphs": sum(sample.label == 0 for sample in samples),
            "attack_graphs": sum(sample.label == 1 for sample in samples),
            "sessions": len(session_set),
            "node_count_quantiles": _quantiles(
                [sample.node_features.shape[0] for sample in samples]
            ),
            "edge_count_quantiles": _quantiles(
                [sample.edge_features.shape[0] for sample in samples]
            ),
            "node_feature_availability": _availability(samples, "node", groups),
            "edge_feature_availability": _availability(samples, "edge", groups),
            "protocol_distribution": protocols,
            "protocol_total_variation_from_train": _total_variation(train_protocols, protocols),
            "attack_categories": dict(sorted(categories.items())),
            "attack_subcategories": dict(sorted(subcategories.items())),
        }

    names = list(graph_splits)
    overlap = {
        f"{left}__{right}": {
            "window_overlap": len(identities[left] & identities[right]),
            "session_overlap": len(sessions[left] & sessions[right]),
        }
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    train_normal = [sample for sample in graph_splits["train"] if sample.label == 0]
    audit_normal = [sample for sample in graph_splits["audit"] if sample.label == 0]
    evaluation_batch_size = int(config["training"]["batch_size"])
    mask_diagnostics = _mask_diagnostics(
        train_normal,
        masker,
        config["audit"]["mask_seeds"],
        groups,
        evaluation_batch_size,
    )
    audit_mask_diagnostics = _mask_diagnostics(
        audit_normal,
        masker,
        config["audit"]["mask_seeds"],
        groups,
        evaluation_batch_size,
    )
    reciprocal_timing = _reciprocal_timing_exact_match(all_samples, groups)
    blockers: list[str] = []
    if any(item["window_overlap"] != 0 for item in overlap.values()):
        blockers.append("split_window_overlap")
    minimum_rate = float(config["audit"]["minimum_mask_target_rate_per_family"])
    for family, rate in mask_diagnostics["primary_scalar_rates"].items():
        if not np.isfinite(rate) or rate < minimum_rate:
            blockers.append(f"insufficient_mask_rate:{family}")
    for family, rate in audit_mask_diagnostics["primary_scalar_rates"].items():
        if not np.isfinite(rate) or rate < minimum_rate:
            blockers.append(f"insufficient_audit_mask_rate:{family}")
    minimum_stratum = int(config["audit"]["minimum_mask_stratum_available_scalars"])
    stratum_counts = mask_diagnostics["available_scalars_by_dependency_degree"]
    stratum_rates = mask_diagnostics["primary_scalar_rates_by_dependency_degree"]
    for stratum, count in stratum_counts.items():
        if count >= minimum_stratum and stratum_rates[stratum] == 0.0:
            blockers.append(f"unmasked_degree_stratum:{stratum}")
    minimum_audit_scalars = int(
        config["audit"].get(
            "minimum_target_scalars_per_degree_stratum_across_masks", 0
        )
    )
    minimum_audit_graphs = int(
        config["audit"].get(
            "minimum_target_graphs_per_degree_stratum_across_masks", 0
        )
    )
    for stratum, available_count in audit_mask_diagnostics[
        "available_scalars_by_dependency_degree"
    ].items():
        if available_count < minimum_stratum:
            continue
        targeted_scalars = audit_mask_diagnostics[
            "targeted_scalars_by_dependency_degree"
        ].get(stratum, 0)
        targeted_graphs = audit_mask_diagnostics[
            "targeted_graphs_by_dependency_degree"
        ].get(stratum, 0)
        if targeted_scalars < minimum_audit_scalars:
            blockers.append(
                f"insufficient_audit_target_scalars:{stratum}:"
                f"{targeted_scalars}<{minimum_audit_scalars}"
            )
        if targeted_graphs < minimum_audit_graphs:
            blockers.append(
                f"insufficient_audit_target_graphs:{stratum}:"
                f"{targeted_graphs}<{minimum_audit_graphs}"
            )
    for split, minimum in config["audit"]["minimum_normal_graphs"].items():
        observed = split_stats[split]["normal_graphs"]
        if observed < int(minimum):
            blockers.append(
                f"insufficient_normal_support:{split}:{observed}<{int(minimum)}"
            )
    test_attacks = split_stats["test"]["attack_graphs"]
    minimum_test_attacks = int(config["audit"]["minimum_test_attack_graphs"])
    if test_attacks < minimum_test_attacks:
        blockers.append(
            f"insufficient_attack_support:test:{test_attacks}<{minimum_test_attacks}"
        )
    maximum_reciprocal_exposure = float(
        config["audit"]["maximum_reciprocal_timing_exact_match_exposure_rate"]
    )
    observed_reciprocal_exposure = float(
        reciprocal_timing["reciprocal_exact_match_exposure_rate"]
    )
    if observed_reciprocal_exposure > maximum_reciprocal_exposure:
        blockers.append(
            "reciprocal_timing_exact_match_exposure_rate:"
            f"{observed_reciprocal_exposure:.6f}>"
            f"{maximum_reciprocal_exposure:.6f}"
        )

    return {
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "ingestion": ingestion.to_dict(),
        "input_sha256": {str(path.resolve()): _sha256(path) for path in input_paths},
        "config_sha256": config_fingerprint(config),
        "splits": split_stats,
        "split_overlap": overlap,
        "deterministic_shortcut_exposure": _shortcut_rates(all_samples, groups),
        "reciprocal_timing": reciprocal_timing,
        "masking": mask_diagnostics,
        "audit_masking": audit_mask_diagnostics,
    }
