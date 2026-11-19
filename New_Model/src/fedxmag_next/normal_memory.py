"""Conservative extension of frozen normal support using verified past nodes.

Normal prototypes can only lower each node energy. Final thresholds are computed
from the ORIGINAL frozen energies, so every retained alarm must also be a frozen
alarm. This is a normality filter, not a claim of exchangeability under drift.
The current window enters memory only after its independent normal verification.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .detector import conformal_threshold
from .experiments import knn_scores, sha256, write_json
from .geometry import ConditionalMahalanobis
from .record_calibration import cluster_difference


def geometry_values(state: dict, values: torch.Tensor) -> np.ndarray:
    support = ConditionalMahalanobis.from_state_dict(state)
    return torch.cat([support.score(part)[0] for part in values.split(4096)]).numpy()


def source_whiten(state: dict, values: torch.Tensor) -> torch.Tensor:
    """Use frozen source covariance to measure distances between normal prototypes."""
    support = ConditionalMahalanobis.from_state_dict(state)
    if support.components != 1:
        raise ValueError("A single frozen source covariance is required for the memory metric")
    return torch.linalg.solve_triangular(
        support.cholesky[0], (values.double() - support.means[0]).T, upper=False
    ).T.contiguous()


def entity_distances(reference, queries, reference_keys, query_keys):
    """Nearest memory of the same entity; unknown entities receive infinite distance."""
    reference_keys = torch.as_tensor(reference_keys)
    query_keys = torch.as_tensor(query_keys)
    result = []
    for begin in range(0, len(queries), 512):
        query = queries[begin : begin + 512].double()
        distances = torch.cdist(query, reference.double())
        eligible = query_keys[begin : begin + 512, None] == reference_keys[None, :]
        distances.masked_fill_(~eligible, torch.inf)
        result.append(distances.min(dim=1).values)
    return torch.cat(result).double()


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    output, path, reference_run = (
        Path(args.output).resolve(),
        Path(args.cache).resolve(),
        Path(args.reference_run).resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    cache = torch.load(path, map_location="cpu", weights_only=True)
    stored = torch.load(reference_run / "scores.pt", map_location="cpu", weights_only=True)
    if not torch.equal(cache["target_minutes"], stored["minutes"]):
        raise ValueError("Cached vectors and frozen decisions cover different windows")
    if not torch.equal(cache["target_labels"], stored["labels"]):
        raise ValueError("Cached vectors and frozen truth differ")
    features = cache["node_features"]
    scope = args.scope
    node_keys = cache.get("node_keys")
    if scope == "entity" and node_keys is None:
        raise ValueError("Personalized memory requires aligned grouping keys")
    artifact = cache["source_scalers"]
    node_minutes = cache["node_minutes"].numpy()
    reference, selection = cache["source_reference"], cache["source_selection"]
    minutes, labels = cache["target_minutes"].numpy(), cache["target_labels"].numpy()
    if int(cache["encoder"]["latest_reference_observed_at"]) >= int(minutes.min()) * 60:
        raise ValueError("Encoder contains target information")
    if np.any(node_minutes[reference.numpy()] >= minutes.min()):
        raise ValueError("Source node geometry contains target information")
    groups = {}
    order = np.argsort(node_minutes, kind="stable")
    boundaries = np.flatnonzero(np.r_[True, np.diff(node_minutes[order]) != 0])
    for start, stop in zip(boundaries, np.r_[boundaries[1:], len(order)], strict=True):
        groups[int(node_minutes[order[start]])] = order[start:stop]
    raw = features["raw"]
    node_models = artifact["node_models"]
    metric = args.metric
    metric_features, metric_supports = features, {}
    raw_metric_reference = artifact["raw_reference"]
    if metric == "source_mahalanobis":
        metric_supports["raw"] = node_models["raw_m1"]
        for name in features:
            if name.startswith("deep_"):
                seed = int(name.split("_")[-1])
                metric_supports[name] = node_models[f"seed_{seed}_m1"]["support"]
        metric_features = {
            name: source_whiten(metric_supports[name], values) for name, values in features.items()
        }
        raw_metric_reference = source_whiten(metric_supports["raw"], raw_metric_reference)
    raw_knn = knn_scores(artifact["raw_reference"], raw).numpy()
    raw_maha = geometry_values(node_models["raw_m8"], raw)
    radii = {
        "raw": max(
            float(
                torch.quantile(
                    knn_scores(raw_metric_reference, metric_features["raw"][selection], k=1), 0.95
                )
            ),
            1e-6,
        )
    }
    global_deep, normalized = {}, []
    deep_names = sorted(name for name in features if name.startswith("deep_"))
    for name in deep_names:
        seed = int(name.split("_")[-1])
        state = node_models[f"seed_{seed}_m8"]
        energy = geometry_values(state["support"], features[name])
        global_deep[name] = energy
        normalized.append((energy - state["center"]) / state["scale"] + 1)
        radii[name] = max(
            float(
                torch.quantile(
                    knn_scores(
                        metric_features[name][reference], metric_features[name][selection], k=1
                    ),
                    0.95,
                )
            ),
            1e-6,
        )
    deep = np.mean(normalized, axis=0)
    fusion = node_models["fusion_m8"]
    frozen = {
        "raw_knn": raw_knn,
        "raw_mahalanobis_m8": raw_maha,
        "deep_mahalanobis_m8_ensemble": deep,
        "raw_plus_deep_m8": np.maximum(raw_knn / fusion["raw"], deep / fusion["deep"]),
    }
    q95 = {
        "raw_knn": max(float(np.quantile(raw_knn[selection], 0.95)), 1e-6),
        "raw_maha": max(float(np.quantile(raw_maha[selection], 0.95)), 1e-6),
        **{
            name: max(float(np.quantile(energy[selection], 0.95)), 1e-6)
            for name, energy in global_deep.items()
        },
    }
    for name, values in frozen.items():
        actual = np.asarray(
            [
                np.sort(values[groups[int(t)]])[-3:].mean() if int(t) in groups else 0.0
                for t in minutes
            ]
        )
        expected = stored["methods"][name]["energy"].numpy()
        if np.max(np.abs(actual - expected) / (1 + np.abs(expected))) > 1e-5:
            raise ValueError(f"Frozen node replay differs: {name}")
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "cache_sha256": sha256(path),
            "frozen_scores_sha256": sha256(reference_run / "scores.pt"),
            "radii": radii,
            "energy_q95": q95,
            "radius_fit": "source selection normals only",
            "memory_metric": metric,
            "memory_scope": scope,
            "guarantee": "energy never increases; frozen calibration stays unchanged",
            "admission": (
                "whole-normal feedback after prediction; identities never numeric inputs, "
                "equality keys only select memory when scope=entity"
            ),
        },
    )
    histories = {
        name: deque(
            stored["methods"][name]["source_calibration"].tolist()[-args.buffer :],
            maxlen=args.buffer,
        )
        for name in frozen
    }
    global_flags = {name: np.zeros(len(minutes), dtype=bool) for name in frozen}
    filtered_flags = {name: np.zeros(len(minutes), dtype=bool) for name in frozen}
    valid = np.zeros(len(minutes), dtype=bool)
    memory = deque(maxlen=args.memory_nodes)
    memory_tensors = {}
    audit = []
    for i, minute in enumerate(minutes):
        ids = groups.get(int(minute))
        if ids is None:
            continue
        baseline = {
            name: float(np.sort(values[ids])[-3:].mean()) for name, values in frozen.items()
        }
        filtered = baseline.copy()
        if memory:
            previous = np.asarray(memory, dtype=int)
            if node_minutes[previous].max() >= minute:
                raise ValueError("Current/future node entered normal memory")
            distances = {}
            for name, values in metric_features.items():
                if scope == "entity":
                    nearest = entity_distances(
                        memory_tensors[name], values[ids], node_keys[previous], node_keys[ids]
                    )
                else:
                    nearest = knn_scores(memory_tensors[name], values[ids], k=1)
                distances[name] = nearest.numpy()
            raw_distance = np.square(distances["raw"] / radii["raw"])
            new_knn = np.minimum(raw_knn[ids], raw_distance * q95["raw_knn"])
            new_maha = np.minimum(raw_maha[ids], raw_distance * q95["raw_maha"])
            new_deep = []
            for name in deep_names:
                seed = int(name.split("_")[-1])
                state = node_models[f"seed_{seed}_m8"]
                local = np.square(distances[name] / radii[name]) * q95[name]
                energy = np.minimum(global_deep[name][ids], local)
                new_deep.append((energy - state["center"]) / state["scale"] + 1)
            new_deep = np.mean(new_deep, axis=0)
            channels = {
                "raw_knn": new_knn,
                "raw_mahalanobis_m8": new_maha,
                "deep_mahalanobis_m8_ensemble": new_deep,
                "raw_plus_deep_m8": np.maximum(new_knn / fusion["raw"], new_deep / fusion["deep"]),
            }
            filtered = {
                name: float(np.sort(values)[-3:].mean()) for name, values in channels.items()
            }
        for name in frozen:
            if filtered[name] > baseline[name] + 1e-6 * (1 + abs(baseline[name])):
                raise ValueError("A normal-memory extension increased anomaly energy")
            threshold = conformal_threshold(histories[name], 0.05)
            if threshold is not None:
                valid[i] = True
                global_flags[name][i] = baseline[name] > threshold
                filtered_flags[name][i] = filtered[name] > threshold
        audit.append({"minute": int(minute), "prior_memory_nodes": len(memory)})
        # A benchmark oracle supplies this verification only AFTER prediction.
        if labels[i] == 0:
            for name in frozen:
                histories[name].append(baseline[name])
            memory.extend(ids.tolist())
            bank = np.asarray(memory, dtype=int)
            memory_tensors = {
                name: values[bank].double() for name, values in metric_features.items()
            }
    results, archive = {}, {}
    for name in frozen:
        if np.any(filtered_flags[name] & ~global_flags[name]):
            raise ValueError("The normality filter created a new alarm")
        for suffix, flags in (("frozen", global_flags[name]), ("memory", filtered_flags[name])):
            method = f"{name}_{suffix}"
            normal, attack = valid & (labels == 0), valid & (labels == 1)
            result = {
                "coverage": float(valid.mean()),
                "tpr": float(flags[attack].mean()),
                "fpr": float(flags[normal].mean()),
                "normal_windows": int(normal.sum()),
                "attack_windows": int(attack.sum()),
                "end_to_end_attack_recall": float(flags[labels == 1].mean()),
            }
            result["per_type_tpr"] = {}
            for kind in sorted({kind for group in cache["target_types"] for kind in group}):
                available = np.asarray([kind in group for group in cache["target_types"]])
                chosen = available & valid
                result["per_type_tpr"][kind] = {
                    "available_windows": int(available.sum()),
                    "evaluated_windows": int(chosen.sum()),
                    "tpr": float(flags[chosen].mean()) if chosen.any() else None,
                }
            results[method] = result
            archive[method] = {"flags": torch.from_numpy(flags), "valid": torch.from_numpy(valid)}
            print(method, {k: result[k] for k in ("tpr", "fpr", "coverage")}, flush=True)
    subset_audit = None
    if args.global_control:
        if scope != "entity":
            raise ValueError("A global comparison is only supported for entity-scoped memory")
        control_path = Path(args.global_control)
        control = torch.load(control_path / "decisions.pt", map_location="cpu", weights_only=True)
        control_registration = json.loads(
            (control_path / "registration.json").read_text()
        )
        if control_registration["memory_metric"] != metric:
            raise ValueError("Global memory comparison uses a different metric")
        for key, value in radii.items():
            if not np.isclose(value, control_registration["radii"][key], rtol=1e-10, atol=1e-12):
                raise ValueError("Global comparison radius differs")
        for name in frozen:
            if not torch.equal(
                control[f"{name}_frozen"]["flags"], archive[f"{name}_frozen"]["flags"]
            ):
                raise ValueError("Global comparison frozen baseline differs")
            if not torch.equal(control[f"{name}_memory"]["valid"], torch.from_numpy(valid)):
                raise ValueError("Global comparison coverage differs")
            if torch.any(control[f"{name}_memory"]["flags"] & ~archive[f"{name}_memory"]["flags"]):
                raise ValueError("Entity-filter alarms must contain all global-filter alarms")
        subset_audit = True
    differences = {
        name: cluster_difference(
            (filtered_flags[name], valid), (global_flags[name], valid), labels, minutes // 60
        )
        for name in frozen
    }
    write_json(
        output / "report.json",
        {
            "status": "past_verified_normal_memory_development",
            "dataset": cache["dataset"],
            "methods": results,
            "new_alarms": 0,
            "memory_scope": scope,
            "global_filter_alarms_subset_of_entity_filter": subset_audit,
            "hour_bootstrap_memory_minus_frozen": differences,
            "memory_audit": audit,
        },
    )
    torch.save(archive, output / "decisions.pt")
    torch.save(
        {
            "node_detector": artifact,
            "normal_memory": memory_tensors,
            "memory_metric": metric,
            "metric_source_supports": metric_supports,
            "memory_scope": scope,
            "memory_node_keys": None
            if node_keys is None
            else node_keys[np.asarray(memory, dtype=np.int64)],
            "radii": radii,
            "energy_q95": q95,
            "frozen_calibration": {name: list(values) for name, values in histories.items()},
        },
        output / "normal_memory_detector.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--buffer", type=int, default=39)
    parser.add_argument("--memory-nodes", type=int, default=5000)
    parser.add_argument(
        "--metric", choices=("euclidean", "source_mahalanobis"), default="euclidean"
    )
    parser.add_argument("--scope", choices=("global", "entity"), default="global")
    parser.add_argument(
        "--global-control", help="Audit alarm containment against a matched global run"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
