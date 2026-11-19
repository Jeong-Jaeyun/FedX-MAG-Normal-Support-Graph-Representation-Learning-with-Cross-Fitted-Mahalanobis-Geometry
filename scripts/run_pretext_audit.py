"""Train M0 across seeds and enforce its held-out admissibility gates."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

from fedxmag.config import (
    config_fingerprint,
    feature_groups_from_config,
    load_stage1_config,
    preprocessing_code_fingerprint,
    training_code_fingerprint,
)
from fedxmag.losses import FAMILY_KEYS
from fedxmag.masking import FamilyMasker
from fedxmag.models import (
    AdjacencyShuffledGraphModel,
    ConstantPredictor,
    LocalPredictor,
    M0GraphModel,
    SetContextPredictor,
)
from fedxmag.serialization import load_graph_samples, schema_fingerprint
from fedxmag.training import evaluate_m0_repeated, train_m0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_m0.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": pstdev(values)}


def _across_model_seeds(runs: list[dict[str, Any]], predictor: str) -> dict[str, Any]:
    keys = runs[0][predictor]["aggregate"]
    result: dict[str, Any] = {
        key: _mean_std([float(run[predictor]["aggregate"][key]) for run in runs])
        for key in keys
        if key.endswith("_mean")
    }
    stratum_keys = sorted(
        {
            key
            for run in runs
            for key in run[predictor].get("degree_strata", {})
        }
    )
    result["degree_strata"] = {
        key: {
            "mse": _mean_std(
                [
                    float(run[predictor]["degree_strata"][key]["mse_mean"])
                    for run in runs
                    if key in run[predictor].get("degree_strata", {})
                ]
            ),
            "scalar_count_mean_across_masks_and_models": mean(
                float(run[predictor]["degree_strata"][key]["scalar_count_mean"])
                for run in runs
                if key in run[predictor].get("degree_strata", {})
            ),
            "graph_count_mean_across_masks_and_models": mean(
                float(run[predictor]["degree_strata"][key]["graph_count_mean"])
                for run in runs
                if key in run[predictor].get("degree_strata", {})
            ),
        }
        for key in stratum_keys
    }
    return result


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _configure_determinism(training: dict[str, Any]) -> None:
    enabled = bool(training["deterministic_algorithms"])
    warn_only = bool(training["deterministic_warn_only"])
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = enabled
        torch.backends.cudnn.benchmark = False


def _runtime_metadata(device: str) -> dict[str, Any]:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "requested_device": device,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_devices": cuda_devices,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only_enabled": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }


def _ensure_output_is_safe(output_dir: Path, names: list[str], overwrite: bool) -> None:
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing artifacts: {rendered}")


def main() -> None:
    args = parse_args()
    config = load_stage1_config(args.config)
    groups = feature_groups_from_config(config)
    expected_config_hash = config_fingerprint(config)
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    preflight = json.loads((args.data_dir / "preflight.json").read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != expected_config_hash:
        raise ValueError("Prepared data and training configuration fingerprints differ")
    if manifest.get("schema_sha256") != schema_fingerprint(groups):
        raise ValueError("Prepared data and runtime feature schemas differ")
    if manifest.get("preprocessing_code_sha256") != preprocessing_code_fingerprint():
        raise ValueError("Prepared data and runtime preprocessing code fingerprints differ")
    if preflight.get("status") != "pass":
        raise ValueError("Training is blocked because preprocessing preflight did not pass")

    train_samples = load_graph_samples(args.data_dir / "train_normal.pt", groups)
    select_samples = load_graph_samples(args.data_dir / "select_normal.pt", groups)
    audit_samples = load_graph_samples(args.data_dir / "audit_normal.pt", groups)
    masking = config["masking"]
    masker = FamilyMasker(
        attribute_budget=float(masking["attribute_budget"]),
        family_mask_probability=float(masking["family_mask_probability"]),
        minimum_visible_per_entity=int(masking["minimum_visible_per_entity"]),
        minimum_visible_per_graph=int(masking["minimum_visible_per_graph"]),
        groups=groups,
    )
    training = config["training"]
    _configure_determinism(training)
    runtime = _runtime_metadata(args.device)
    model_config = config["model"]
    loss_weights = {key: float(config["loss"][key]) for key in FAMILY_KEYS}
    model_seeds = [int(seed) for seed in config["experiment"]["seeds"]]
    mask_seeds = [int(seed) for seed in config["audit"]["mask_seeds"]]
    validation_mask_seeds = [
        int(seed) for seed in training["validation_mask_seeds"]
    ]
    selection_metric = str(training["selection_metric"])
    shuffle_seed = int(config["controls"]["adjacency_shuffle_seed"])
    endpoint_shuffle_mode = str(
        config["controls"].get("endpoint_shuffle_mode", "destination_only")
    )

    common_model_kwargs = {
        "node_dim": groups.node_dim,
        "edge_dim": groups.edge_dim,
        "hidden_dim": int(model_config["hidden_dim"]),
        "graph_dim": int(model_config["graph_dim"]),
        "num_layers": int(model_config["num_layers"]),
        "dropout": float(model_config["dropout"]),
    }
    model_builders = {
        "local": lambda: LocalPredictor(
            groups.node_dim,
            groups.edge_dim,
            hidden_dim=int(model_config["hidden_dim"]),
            dropout=float(model_config["dropout"]),
        ),
        "set_context": lambda: SetContextPredictor(**common_model_kwargs),
        "shuffled_graph": lambda: AdjacencyShuffledGraphModel(
            **common_model_kwargs,
            shuffle_seed=shuffle_seed,
            endpoint_shuffle_mode=endpoint_shuffle_mode,
        ),
        "graph": lambda: M0GraphModel(**common_model_kwargs),
    }
    parameter_counts = {
        name: _parameter_count(builder()) for name, builder in model_builders.items()
    }
    if parameter_counts["set_context"] != parameter_counts["graph"]:
        raise AssertionError("Set-context control is not exactly capacity-matched")
    if parameter_counts["shuffled_graph"] != parameter_counts["graph"]:
        raise AssertionError("Shuffled-graph control is not exactly capacity-matched")
    capacity_ratios = {
        "graph_over_local": parameter_counts["graph"] / parameter_counts["local"],
        "graph_over_set_context": (
            parameter_counts["graph"] / parameter_counts["set_context"]
        ),
        "graph_over_shuffled_graph": (
            parameter_counts["graph"] / parameter_counts["shuffled_graph"]
        ),
    }

    artifact_names = ["pretext_audit.json"] + [
        f"m0_{predictor}_seed_{seed}.pt"
        for predictor in model_builders
        for seed in model_seeds
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_output_is_safe(args.output_dir, artifact_names, args.overwrite)

    constant_metrics = evaluate_m0_repeated(
        ConstantPredictor(),
        audit_samples,
        masker,
        mask_seeds,
        batch_size=int(training["batch_size"]),
        device=args.device,
        loss_weights=loss_weights,
        groups=groups,
    )
    seed_runs: list[dict[str, Any]] = []
    for seed in model_seeds:
        seed_run: dict[str, Any] = {
            "seed": seed,
            "epochs_run": {},
            "best_select_metric": {},
            "selection_metric": selection_metric,
        }
        for predictor, builder in model_builders.items():
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
                validation_mask_seeds=validation_mask_seeds,
                selection_metric=selection_metric,
                loss_weights=loss_weights,
                seed=seed,
                device=args.device,
                groups=groups,
            )
            metrics = evaluate_m0_repeated(
                model,
                audit_samples,
                masker,
                mask_seeds,
                batch_size=int(training["batch_size"]),
                device=args.device,
                loss_weights=loss_weights,
                groups=groups,
            )
            seed_run[predictor] = metrics
            seed_run["epochs_run"][predictor] = len(history)
            seed_run["best_select_metric"][predictor] = min(
                float(item["selection_metric_value"]) for item in history
            )
            torch.save(
                {
                    "model": {
                        key: value.cpu() for key, value in model.state_dict().items()
                    },
                    "model_type": predictor,
                    "parameter_count": parameter_counts[predictor],
                    "node_features": groups.node_names,
                    "edge_features": groups.edge_names,
                    "schema_sha256": schema_fingerprint(groups),
                    "config_sha256": expected_config_hash,
                    "preprocessing_code_sha256": preprocessing_code_fingerprint(),
                    "training_code_sha256": training_code_fingerprint(),
                    "selection_metric": selection_metric,
                    "runtime": runtime,
                    "seed": seed,
                    "stage": "M0",
                },
                args.output_dir / f"m0_{predictor}_seed_{seed}.pt",
            )
        seed_runs.append(seed_run)

    across = {
        predictor: _across_model_seeds(seed_runs, predictor)
        for predictor in model_builders
    }
    local_macro = across["local"]["macro_family_mse_mean"]["mean"]
    set_macro = across["set_context"]["macro_family_mse_mean"]["mean"]
    shuffled_macro = across["shuffled_graph"]["macro_family_mse_mean"]["mean"]
    graph_macro = across["graph"]["macro_family_mse_mean"]["mean"]
    constant_macro = constant_metrics["aggregate"]["macro_family_mse_mean"]
    audit_config = config["audit"]
    minimum_local_gain = float(audit_config["minimum_graph_gain_over_local"])
    minimum_set_gain = float(audit_config["minimum_graph_gain_over_set_context"])
    minimum_shuffled_gain = float(
        audit_config["minimum_graph_gain_over_shuffled_graph"]
    )
    maximum_mse = float(audit_config["maximum_standardized_graph_mse"])
    family_criteria: dict[str, Any] = {}
    for key in FAMILY_KEYS:
        local_value = across["local"][f"{key}_mse_mean"]["mean"]
        set_value = across["set_context"][f"{key}_mse_mean"]["mean"]
        shuffled_value = across["shuffled_graph"][f"{key}_mse_mean"]["mean"]
        graph_value = across["graph"][f"{key}_mse_mean"]["mean"]
        constant_value = constant_metrics["aggregate"][f"{key}_mse_mean"]
        local_gain = (local_value - graph_value) / local_value
        set_gain = (set_value - graph_value) / set_value
        shuffled_gain = (shuffled_value - graph_value) / shuffled_value
        family_criteria[key] = {
            "local_mse": local_value,
            "set_context_mse": set_value,
            "shuffled_graph_mse": shuffled_value,
            "graph_mse": graph_value,
            "constant_mse": constant_value,
            "graph_gain_over_local": local_gain,
            "graph_gain_over_set_context": set_gain,
            "graph_gain_over_shuffled_graph": shuffled_gain,
            "minimum_graph_gain_over_local": minimum_local_gain,
            "minimum_graph_gain_over_set_context": minimum_set_gain,
            "minimum_graph_gain_over_shuffled_graph": minimum_shuffled_gain,
            "maximum_standardized_graph_mse": maximum_mse,
            "local_gain_pass": local_gain >= minimum_local_gain,
            "set_context_gain_pass": set_gain >= minimum_set_gain,
            "shuffled_graph_gain_pass": shuffled_gain >= minimum_shuffled_gain,
            "constant_pass": graph_value < constant_value,
            "predictability_pass": graph_value <= maximum_mse,
        }
        if not bool(audit_config["require_graph_better_than_constant"]):
            family_criteria[key]["constant_pass"] = True
        family_criteria[key]["pass"] = all(
            family_criteria[key][criterion]
            for criterion in (
                "local_gain_pass",
                "set_context_gain_pass",
                "shuffled_graph_gain_pass",
                "constant_pass",
                "predictability_pass",
            )
        )
    criteria = {
        "macro_summary_not_a_gate": {
            "graph_gain_over_local": (local_macro - graph_macro) / local_macro,
            "graph_gain_over_set_context": (set_macro - graph_macro) / set_macro,
            "graph_gain_over_shuffled_graph": (
                (shuffled_macro - graph_macro) / shuffled_macro
            ),
            "graph": graph_macro,
            "local": local_macro,
            "set_context": set_macro,
            "shuffled_graph": shuffled_macro,
            "constant": constant_macro,
            "note": "Descriptive only; every family must independently pass.",
        },
        "families": family_criteria,
    }
    passed = all(bool(item["pass"]) for item in family_criteria.values())
    report = {
        "status": "pass" if passed else "fail",
        "dataset_role": "normal audit only; select used only for early stopping",
        "preflight_audit_masking": preflight.get("audit_masking"),
        "model_seeds": model_seeds,
        "shared_evaluation_mask_seeds": mask_seeds,
        "selection_metric": selection_metric,
        "runtime": runtime,
        "model_parameter_counts": parameter_counts,
        "capacity_ratios": capacity_ratios,
        "adjacency_shuffle_seed": shuffle_seed,
        "endpoint_shuffle_mode": endpoint_shuffle_mode,
        "constant": constant_metrics,
        "per_model_seed": seed_runs,
        "across_model_seeds": across,
        "criteria": criteria,
        "config_sha256": expected_config_hash,
        "preprocessing_code_sha256": preprocessing_code_fingerprint(),
        "training_code_sha256": training_code_fingerprint(),
        "schema_sha256": schema_fingerprint(groups),
    }
    (args.output_dir / "pretext_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError("M0 admissibility audit failed; see pretext_audit.json")


if __name__ == "__main__":
    main()
