"""Pre-registered M0 confirmation: protocol binding, gates, and paired analyses.

This module deliberately lives outside ``models/`` and ``training/`` so that
adding it does not change the frozen v3b training-code fingerprint. Gate values
are always computed from the frozen :func:`evaluate_m0_repeated` path; the
per-graph path below only feeds descriptive paired statistics and is asserted to
reproduce the frozen aggregates exactly.
"""

from __future__ import annotations

import hashlib
import platform
import time
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from fedxmag.config import (
    config_fingerprint,
    feature_groups_from_config,
    load_stage1_config,
    preprocessing_code_fingerprint,
    training_code_fingerprint,
)
from fedxmag.losses import FAMILY_KEYS
from fedxmag.masking import FamilyMasker, MaskedGraphBatch
from fedxmag.models import (
    AdjacencyShuffledGraphModel,
    ConstantPredictor,
    LocalPredictor,
    M0GraphModel,
    SetContextPredictor,
)
from fedxmag.preflight import _mask_diagnostics
from fedxmag.schema import FeatureGroups, GraphSample, collate_graphs
from fedxmag.serialization import schema_fingerprint
from fedxmag.training import evaluate_m0_repeated, train_m0

CONTROLS = ("local", "set_context", "shuffled_graph")
PREDICTORS = (*CONTROLS, "graph")
GATE_KEYS = (
    "local_gain_pass",
    "set_context_gain_pass",
    "shuffled_graph_gain_pass",
    "constant_pass",
    "predictability_pass",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Load the protocol and assert it agrees with its frozen base configuration."""

    root = Path(project_root)
    with Path(path).open("r", encoding="utf-8") as stream:
        protocol = yaml.safe_load(stream)
    frozen = protocol["frozen_inputs"]
    config = load_stage1_config(root / frozen["base_config"])
    groups = feature_groups_from_config(config)
    checks = {
        "base_config_sha256": config_fingerprint(config),
        "base_config_file_sha256": sha256_file(root / frozen["base_config"]),
        "schema_sha256": schema_fingerprint(groups),
        "preprocessing_code_sha256": preprocessing_code_fingerprint(root),
        "training_code_sha256": training_code_fingerprint(root),
    }
    mismatched = {key: value for key, value in checks.items() if frozen[key] != value}
    if mismatched:
        raise ValueError(f"Frozen fingerprints differ from the registered protocol: {mismatched}")
    if [int(seed) for seed in protocol["model_seeds"]] != [
        int(seed) for seed in config["experiment"]["seeds"]
    ]:
        raise ValueError("Protocol model seeds differ from the base configuration")
    if [int(seed) for seed in protocol["confirmation_mask_seeds"]] != [
        int(seed) for seed in config["audit"]["mask_seeds"]
    ]:
        raise ValueError("Protocol confirmation masks differ from the base configuration")
    overlap = set(protocol["confirmation_mask_seeds"]) & set(
        config["training"]["validation_mask_seeds"]
    )
    if overlap:
        raise ValueError(f"Confirmation and early-stopping masks overlap: {sorted(overlap)}")
    for key, value in protocol["gates"].items():
        if config["audit"][key] != value:
            raise ValueError(f"Protocol gate {key} differs from the base configuration")
    if tuple(protocol["predictors"]) != PREDICTORS:
        raise ValueError("Protocol predictors must be exactly the frozen M0 control ladder")
    roles = protocol["data"]["roles"]
    forbidden = set(protocol["data"]["never_opened_by_this_protocol"])
    if forbidden & set(roles.values()):
        raise ValueError("A protocol role uses a forbidden artifact")
    prepared = root / protocol["data"]["prepared_dir"]
    for name, expected in protocol["data"]["file_sha256"].items():
        if name in forbidden:
            raise ValueError(f"Forbidden artifact {name} cannot be bound by hash")
        observed = sha256_file(prepared / name)
        if observed != expected:
            raise ValueError(f"Prepared artifact {name} differs from the registered hash")
    return {"protocol": protocol, "config": config, "groups": groups}


def masker_from_config(config: Mapping[str, Any], groups: FeatureGroups) -> FamilyMasker:
    masking = config["masking"]
    return FamilyMasker(
        attribute_budget=float(masking["attribute_budget"]),
        family_mask_probability=float(masking["family_mask_probability"]),
        minimum_visible_per_entity=int(masking["minimum_visible_per_entity"]),
        minimum_visible_per_graph=int(masking["minimum_visible_per_graph"]),
        groups=groups,
    )


def model_builders(config: Mapping[str, Any], groups: FeatureGroups) -> dict[str, Any]:
    """Exactly the run_pretext_audit.py control ladder."""

    model_config = config["model"]
    common = {
        "node_dim": groups.node_dim,
        "edge_dim": groups.edge_dim,
        "hidden_dim": int(model_config["hidden_dim"]),
        "graph_dim": int(model_config["graph_dim"]),
        "num_layers": int(model_config["num_layers"]),
        "dropout": float(model_config["dropout"]),
    }
    shuffle_seed = int(config["controls"]["adjacency_shuffle_seed"])
    shuffle_mode = str(config["controls"].get("endpoint_shuffle_mode", "destination_only"))
    return {
        "local": lambda: LocalPredictor(
            groups.node_dim,
            groups.edge_dim,
            hidden_dim=int(model_config["hidden_dim"]),
            dropout=float(model_config["dropout"]),
        ),
        "set_context": lambda: SetContextPredictor(**common),
        "shuffled_graph": lambda: AdjacencyShuffledGraphModel(
            **common,
            shuffle_seed=shuffle_seed,
            endpoint_shuffle_mode=shuffle_mode,
        ),
        "graph": lambda: M0GraphModel(**common),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def runtime_metadata(device: str) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "requested_device": device,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only_enabled": torch.is_deterministic_algorithms_warn_only_enabled(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }


def support_check(
    samples: Sequence[GraphSample],
    masker: FamilyMasker,
    mask_seeds: Sequence[int],
    groups: FeatureGroups,
    batch_size: int,
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    """Model-free, label-free support check mirroring the audit-role preflight."""

    diagnostics = _mask_diagnostics(samples, masker, mask_seeds, groups, batch_size)
    blockers: list[str] = []
    minimum_graphs = int(requirements["minimum_confirmation_normal_graphs"])
    if len(samples) < minimum_graphs:
        blockers.append(f"insufficient_normal_graphs:{len(samples)}<{minimum_graphs}")
    if any(sample.label != 0 for sample in samples):
        blockers.append("confirmation_partition_contains_attack_graph")
    minimum_rate = float(requirements["minimum_mask_target_rate_per_family"])
    for family, rate in diagnostics["primary_scalar_rates"].items():
        if not np.isfinite(rate) or rate < minimum_rate:
            blockers.append(f"insufficient_mask_rate:{family}")
    minimum_stratum = int(requirements["minimum_mask_stratum_available_scalars"])
    minimum_scalars = int(requirements["minimum_target_scalars_per_degree_stratum_across_masks"])
    minimum_target_graphs = int(
        requirements["minimum_target_graphs_per_degree_stratum_across_masks"]
    )
    for stratum, available in diagnostics["available_scalars_by_dependency_degree"].items():
        if available < minimum_stratum:
            continue
        scalars = diagnostics["targeted_scalars_by_dependency_degree"].get(stratum, 0)
        graphs = diagnostics["targeted_graphs_by_dependency_degree"].get(stratum, 0)
        if scalars < minimum_scalars:
            blockers.append(f"insufficient_target_scalars:{stratum}:{scalars}<{minimum_scalars}")
        if graphs < minimum_target_graphs:
            blockers.append(
                f"insufficient_target_graphs:{stratum}:{graphs}<{minimum_target_graphs}"
            )
    return {
        "status": "pass" if not blockers else "blocked",
        "blockers": blockers,
        "normal_graphs": len(samples),
        "sessions": len({str(sample.metadata["session_key"]) for sample in samples}),
        "mask_diagnostics": diagnostics,
    }


def family_gate_decision(
    constant_aggregate: Mapping[str, float],
    per_seed_aggregates: Mapping[str, Sequence[Mapping[str, float]]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen family-wise gates exactly as run_pretext_audit.py does.

    ``per_seed_aggregates[predictor]`` holds one ``evaluate_m0_repeated``
    aggregate per model seed. Family MSE is the mean of the per-seed mask means.
    """

    families: dict[str, Any] = {}
    for key in FAMILY_KEYS:
        values = {
            predictor: mean(
                float(aggregate[f"{key}_mse_mean"])
                for aggregate in per_seed_aggregates[predictor]
            )
            for predictor in PREDICTORS
        }
        graph = values["graph"]
        constant = float(constant_aggregate[f"{key}_mse_mean"])
        gains = {
            control: (values[control] - graph) / values[control] for control in CONTROLS
        }
        entry: dict[str, Any] = {
            "constant_mse": constant,
            **{f"{predictor}_mse": values[predictor] for predictor in PREDICTORS},
            **{f"graph_gain_over_{control}": gains[control] for control in CONTROLS},
            "local_gain_pass": gains["local"] >= float(gates["minimum_graph_gain_over_local"]),
            "set_context_gain_pass": gains["set_context"]
            >= float(gates["minimum_graph_gain_over_set_context"]),
            "shuffled_graph_gain_pass": gains["shuffled_graph"]
            >= float(gates["minimum_graph_gain_over_shuffled_graph"]),
            "constant_pass": graph < constant
            if bool(gates["require_graph_better_than_constant"])
            else True,
            "predictability_pass": graph <= float(gates["maximum_standardized_graph_mse"]),
        }
        entry["pass"] = all(bool(entry[gate]) for gate in GATE_KEYS)
        families[key] = entry
    return {
        "status": "pass" if all(entry["pass"] for entry in families.values()) else "fail",
        "families": families,
    }


def build_masked_batches(
    samples: Sequence[GraphSample],
    masker: FamilyMasker,
    seed: int,
    batch_size: int,
    groups: FeatureGroups,
) -> list[MaskedGraphBatch]:
    """Reproduce the exact mask sequence that ``evaluate_m0`` draws for ``seed``."""

    generator = torch.Generator().manual_seed(seed)
    batches: list[MaskedGraphBatch] = []
    for start in range(0, len(samples), batch_size):
        batch = collate_graphs(list(samples[start : start + batch_size]), groups)
        batches.append(masker(batch, generator=generator))
    return batches


@torch.no_grad()
def per_graph_family_errors(
    model: nn.Module,
    masked_batches: Sequence[MaskedGraphBatch],
    groups: FeatureGroups,
) -> np.ndarray:
    """Return ``[graphs, families]`` per-graph masked MSE, NaN where untargeted."""

    model.eval()
    rows: list[np.ndarray] = []
    for masked in masked_batches:
        output = model(masked)
        batch = masked.batch
        result = np.full((batch.num_graphs, len(FAMILY_KEYS)), np.nan, dtype=np.float64)
        for column, key in enumerate(FAMILY_KEYS):
            kind, family = key.split("_", 1)
            indices = list(groups.indices(kind, family))
            prediction = getattr(output, f"{kind}_prediction")[:, indices].double()
            target = getattr(masked, f"{kind}_targets")[:, indices].double()
            selected = getattr(masked, f"{kind}_targeted")[:, indices]
            graph_index = getattr(batch, f"{kind}_graph_index")
            squared = ((prediction - target).square() * selected).sum(dim=1)
            count = selected.sum(dim=1).double()
            sums = torch.zeros(batch.num_graphs, dtype=torch.float64).index_add_(
                0, graph_index, squared
            )
            counts = torch.zeros(batch.num_graphs, dtype=torch.float64).index_add_(
                0, graph_index, count
            )
            present = counts > 0
            values = np.full(batch.num_graphs, np.nan)
            values[present.numpy()] = (sums[present] / counts[present]).numpy()
            result[:, column] = values
        rows.append(result)
    return np.concatenate(rows, axis=0)


def graph_balanced_means(errors: np.ndarray) -> np.ndarray:
    """Graph-balanced family MSE from per-graph errors (mean over targeted graphs)."""

    return np.nanmean(errors, axis=0)


def cluster_bootstrap(
    errors: Mapping[str, np.ndarray],
    constant_errors: np.ndarray,
    clusters: Sequence[str],
    resamples: int,
    seed: int,
    interval: float,
) -> dict[str, Any]:
    """Cluster bootstrap of family MSE and Graph gains.

    ``errors[predictor]`` has shape ``[seeds, masks, graphs, families]`` and
    ``constant_errors`` has shape ``[masks, graphs, families]``. Whole clusters
    (capture dates) are resampled so within-date window dependence is retained.
    The statistic mirrors the gate: graph-balanced mean per (seed, mask), then
    averaged over masks and seeds.
    """

    labels, cluster_index = np.unique(np.asarray(clusters), return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(labels), np.full(len(labels), 1.0 / len(labels)), size=resamples
    )
    # Matrix products use torch: the local numpy BLAS build is not usable.
    weights = torch.from_numpy(counts[:, cluster_index].astype(np.float64))

    def statistic(array: np.ndarray) -> np.ndarray:
        present = np.isfinite(array)
        filled = np.where(present, array, 0.0)
        graphs = array.shape[-2]
        flat_values = torch.from_numpy(
            np.ascontiguousarray(np.moveaxis(filled, -2, 0).reshape(graphs, -1))
        )
        flat_present = torch.from_numpy(
            np.ascontiguousarray(
                np.moveaxis(present, -2, 0).reshape(graphs, -1).astype(np.float64)
            )
        )
        numerator = (weights @ flat_values).numpy()
        denominator = (weights @ flat_present).numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = numerator / denominator
        leading = array.shape[:-2] + array.shape[-1:]
        ratio = ratio.reshape((resamples, *leading))
        with warnings.catch_warnings():
            # A resample can contain no targeted graph for a (seed, mask, family).
            warnings.simplefilter("ignore", category=RuntimeWarning)
            while ratio.ndim > 2:
                ratio = np.nanmean(ratio, axis=1)
        return ratio

    lower = (1.0 - interval) / 2.0
    upper = 1.0 - lower
    replicate_means = {name: statistic(array) for name, array in errors.items()}
    replicate_means["constant"] = statistic(constant_errors)
    summary: dict[str, Any] = {"clusters": len(labels), "resamples": resamples}
    for column, key in enumerate(FAMILY_KEYS):
        family: dict[str, Any] = {}
        graph = replicate_means["graph"][:, column]
        for control in (*CONTROLS, "constant"):
            reference = replicate_means[control][:, column]
            gain = (reference - graph) / reference
            family[f"graph_gain_over_{control}"] = {
                "lower": float(np.nanquantile(gain, lower)),
                "upper": float(np.nanquantile(gain, upper)),
                "probability_gain_below_5_percent": float(np.nanmean(gain < 0.05)),
                "probability_gain_not_positive": float(np.nanmean(gain <= 0.0)),
            }
        family["graph_mse"] = {
            "lower": float(np.nanquantile(graph, lower)),
            "upper": float(np.nanquantile(graph, upper)),
        }
        summary[key] = family
    return summary


def per_mask_wins(
    per_seed_runs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Count paired (seed, mask) comparisons in which Graph has lower MSE."""

    result: dict[str, Any] = {}
    for key in FAMILY_KEYS:
        family: dict[str, Any] = {}
        for control in CONTROLS:
            wins = 0
            total = 0
            margins: list[float] = []
            for graph_run, control_run in zip(
                per_seed_runs["graph"], per_seed_runs[control], strict=True
            ):
                for graph_mask, control_mask in zip(
                    graph_run["per_mask"], control_run["per_mask"], strict=True
                ):
                    graph_value = float(graph_mask[f"{key}_mse"])
                    control_value = float(control_mask[f"{key}_mse"])
                    total += 1
                    wins += int(graph_value < control_value)
                    margins.append((control_value - graph_value) / control_value)
            family[control] = {
                "wins": wins,
                "comparisons": total,
                "minimum_relative_margin": min(margins),
                "median_relative_margin": float(np.median(margins)),
            }
        result[key] = family
    return result


def per_seed_gains(
    per_seed_aggregates: Mapping[str, Sequence[Mapping[str, float]]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, seed in enumerate(seeds):
        entry: dict[str, Any] = {}
        for key in FAMILY_KEYS:
            graph = float(per_seed_aggregates["graph"][index][f"{key}_mse_mean"])
            entry[key] = {
                "graph_mse": graph,
                **{
                    f"graph_gain_over_{control}": (
                        float(per_seed_aggregates[control][index][f"{key}_mse_mean"]) - graph
                    )
                    / float(per_seed_aggregates[control][index][f"{key}_mse_mean"])
                    for control in CONTROLS
                },
            }
        result[str(seed)] = entry
    return result


def leave_one_seed_out(
    constant_aggregate: Mapping[str, float],
    per_seed_aggregates: Mapping[str, Sequence[Mapping[str, float]]],
    seeds: Sequence[int],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Descriptive sensitivity: the frozen rule on every subset missing one seed."""

    result: dict[str, Any] = {}
    for index, seed in enumerate(seeds):
        subset = {
            predictor: [
                aggregate
                for position, aggregate in enumerate(values)
                if position != index
            ]
            for predictor, values in per_seed_aggregates.items()
        }
        decision = family_gate_decision(constant_aggregate, subset, gates)
        result[f"without_{seed}"] = {
            "status": decision["status"],
            "failed_families": [
                key for key, entry in decision["families"].items() if not entry["pass"]
            ],
        }
    return result


def train_predictors_for_seed(
    builders: Mapping[str, Any],
    train_samples: Sequence[GraphSample],
    select_samples: Sequence[GraphSample],
    masker: FamilyMasker,
    config: Mapping[str, Any],
    seed: int,
    groups: FeatureGroups,
    device: str = "cpu",
    on_model_done: Any = None,
) -> tuple[dict[str, nn.Module], dict[str, Any]]:
    """Train every predictor for one seed exactly as run_pretext_audit.py does."""

    training = config["training"]
    models: dict[str, nn.Module] = {}
    runs: dict[str, Any] = {}
    for name, builder in builders.items():
        started = time.perf_counter()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = builder()
        history = train_m0(
            model,
            train_samples,
            select_samples,
            masker,
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            early_stopping_patience=int(training["early_stopping_patience"]),
            validation_mask_seeds=tuple(int(item) for item in training["validation_mask_seeds"]),
            selection_metric=str(training["selection_metric"]),
            loss_weights={key: float(config["loss"][key]) for key in FAMILY_KEYS},
            seed=seed,
            device=device,
            groups=groups,
        )
        values = [float(item["selection_metric_value"]) for item in history]
        runs[name] = {
            "epochs_run": len(history),
            "best_epoch": int(np.argmin(values)) + 1,
            "best_select_metric": min(values),
            "wall_seconds": time.perf_counter() - started,
            "history": history,
        }
        models[name] = model
        if on_model_done is not None:
            on_model_done(name, model, runs[name])
    return models, runs


def evaluate_confirmation(
    models: Mapping[tuple[str, int], nn.Module],
    samples: Sequence[GraphSample],
    masker: FamilyMasker,
    seeds: Sequence[int],
    mask_seeds: Sequence[int],
    gates: Mapping[str, Any],
    secondary: Mapping[str, Any],
    batch_size: int,
    groups: FeatureGroups,
    loss_weights: Mapping[str, float] | None = None,
    device: str = "cpu",
    on_decision: Any = None,
) -> dict[str, Any]:
    """Compute the frozen decision first, then the descriptive paired analyses."""

    def frozen(model: nn.Module) -> dict[str, Any]:
        return evaluate_m0_repeated(
            model,
            samples,
            masker,
            mask_seeds,
            batch_size=batch_size,
            device=device,
            loss_weights=loss_weights,
            groups=groups,
        )

    constant = frozen(ConstantPredictor())
    per_seed_runs = {
        name: [frozen(models[(name, seed)]) for seed in seeds] for name in PREDICTORS
    }
    per_seed_aggregates = {
        name: [run["aggregate"] for run in runs] for name, runs in per_seed_runs.items()
    }
    decision = family_gate_decision(constant["aggregate"], per_seed_aggregates, gates)
    if on_decision is not None:
        on_decision(decision)

    cached = [
        build_masked_batches(samples, masker, seed, batch_size, groups) for seed in mask_seeds
    ]
    constant_errors = np.stack(
        [per_graph_family_errors(ConstantPredictor(), batches, groups) for batches in cached]
    )
    errors = {
        name: np.stack(
            [
                np.stack(
                    [
                        per_graph_family_errors(models[(name, seed)], batches, groups)
                        for batches in cached
                    ]
                )
                for seed in seeds
            ]
        )
        for name in PREDICTORS
    }
    worst = 0.0
    for name, runs in (("constant", [constant]), *per_seed_runs.items()):
        array = constant_errors[None] if name == "constant" else errors[name]
        for seed_index, run in enumerate(runs):
            for mask_index, frozen_mask in enumerate(run["per_mask"]):
                recomputed = graph_balanced_means(array[seed_index, mask_index])
                for column, key in enumerate(FAMILY_KEYS):
                    reference = float(frozen_mask[f"{key}_mse"])
                    worst = max(
                        worst, abs(recomputed[column] - reference) / max(abs(reference), 1e-12)
                    )
    if worst > 1e-5:
        raise AssertionError(f"Per-graph analysis does not reproduce frozen aggregates: {worst}")

    clusters = [str(sample.metadata["session_key"]) for sample in samples]
    bootstrap = cluster_bootstrap(
        errors,
        constant_errors,
        clusters,
        int(secondary["bootstrap_resamples"]),
        int(secondary["bootstrap_seed"]),
        float(secondary["interval"]),
    )
    return {
        "decision": decision,
        "constant": constant,
        "per_seed_runs": per_seed_runs,
        "errors": errors,
        "constant_errors": constant_errors,
        "clusters": clusters,
        "secondary_descriptive": {
            "note": "Descriptive only; the decision is fixed by the frozen gates.",
            "per_seed_gains": per_seed_gains(per_seed_aggregates, seeds),
            "per_mask_wins": per_mask_wins(per_seed_runs),
            "leave_one_seed_out": leave_one_seed_out(
                constant["aggregate"], per_seed_aggregates, seeds, gates
            ),
            "degree_strata": degree_strata_across_seeds(
                {"constant": [constant], **per_seed_runs}
            ),
            "session_cluster_bootstrap": bootstrap,
            "per_graph_reproduction_max_relative_discrepancy": worst,
        },
    }


def degree_strata_across_seeds(
    per_seed_runs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for predictor, runs in per_seed_runs.items():
        keys = sorted({key for run in runs for key in run.get("degree_strata", {})})
        result[predictor] = {
            key: {
                "mse_mean": mean(
                    float(run["degree_strata"][key]["mse_mean"])
                    for run in runs
                    if key in run["degree_strata"]
                ),
                "mse_std_across_seeds": pstdev(
                    [
                        float(run["degree_strata"][key]["mse_mean"])
                        for run in runs
                        if key in run["degree_strata"]
                    ]
                ),
                "scalar_count_mean": mean(
                    float(run["degree_strata"][key]["scalar_count_mean"])
                    for run in runs
                    if key in run["degree_strata"]
                ),
                "graph_count_mean": mean(
                    float(run["degree_strata"][key]["graph_count_mean"])
                    for run in runs
                    if key in run["degree_strata"]
                ),
            }
            for key in keys
        }
    return result
