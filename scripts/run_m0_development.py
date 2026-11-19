"""Train-only/select-only M0 development pilot that cannot admit families to M1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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

DEVELOPMENT_REPORT_MASK_SEEDS = (
    300_007,
    300_017,
    300_023,
    300_043,
    300_073,
    300_089,
    300_109,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    config = load_stage1_config(args.config)
    groups = feature_groups_from_config(config)
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_sha256"] != config_fingerprint(config):
        raise ValueError("Prepared data and development configuration differ")
    if manifest["schema_sha256"] != schema_fingerprint(groups):
        raise ValueError("Prepared data and development schema differ")
    if manifest["preprocessing_code_sha256"] != preprocessing_code_fingerprint():
        raise ValueError("Prepared data and preprocessing source differ")

    # Deliberately load only the development roles. Audit, calibration, and test
    # artifacts are neither opened nor evaluated by this entry point.
    train_samples = load_graph_samples(args.data_dir / "train_normal.pt", groups)
    select_samples = load_graph_samples(args.data_dir / "select_normal.pt", groups)
    masking = config["masking"]
    masker = FamilyMasker(
        attribute_budget=float(masking["attribute_budget"]),
        family_mask_probability=float(masking["family_mask_probability"]),
        minimum_visible_per_entity=int(masking["minimum_visible_per_entity"]),
        minimum_visible_per_graph=int(masking["minimum_visible_per_graph"]),
        groups=groups,
    )
    training = config["training"]
    torch.use_deterministic_algorithms(
        bool(training["deterministic_algorithms"]),
        warn_only=bool(training["deterministic_warn_only"]),
    )
    model_config = config["model"]
    common: dict[str, Any] = {
        "node_dim": groups.node_dim,
        "edge_dim": groups.edge_dim,
        "hidden_dim": int(model_config["hidden_dim"]),
        "graph_dim": int(model_config["graph_dim"]),
        "num_layers": int(model_config["num_layers"]),
        "dropout": float(model_config["dropout"]),
    }
    shuffle_seed = int(config["controls"]["adjacency_shuffle_seed"])
    shuffle_mode = str(config["controls"].get("endpoint_shuffle_mode", "destination_only"))
    builders = {
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
    parameter_counts = {name: _parameter_count(builder()) for name, builder in builders.items()}
    if parameter_counts["set_context"] != parameter_counts["graph"]:
        raise AssertionError("Set-context control is not capacity-matched")
    if parameter_counts["shuffled_graph"] != parameter_counts["graph"]:
        raise AssertionError("Shuffled-graph control is not capacity-matched")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"development_select_seed_{args.seed}.json"
    checkpoint_paths = {
        name: args.output_dir / f"development_{name}_seed_{args.seed}.pt"
        for name in builders
    }
    existing = [path for path in (report_path, *checkpoint_paths.values()) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite development artifacts: {existing}")

    loss_weights = {key: float(config["loss"][key]) for key in FAMILY_KEYS}
    constant = evaluate_m0_repeated(
        ConstantPredictor(),
        select_samples,
        masker,
        DEVELOPMENT_REPORT_MASK_SEEDS,
        batch_size=int(training["batch_size"]),
        device=args.device,
        loss_weights=loss_weights,
        groups=groups,
    )
    runs: dict[str, Any] = {}
    for name, builder in builders.items():
        torch.manual_seed(args.seed)
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
            validation_mask_seeds=tuple(training["validation_mask_seeds"]),
            selection_metric=str(training["selection_metric"]),
            loss_weights=loss_weights,
            seed=args.seed,
            device=args.device,
            groups=groups,
        )
        metrics = evaluate_m0_repeated(
            model,
            select_samples,
            masker,
            DEVELOPMENT_REPORT_MASK_SEEDS,
            batch_size=int(training["batch_size"]),
            device=args.device,
            loss_weights=loss_weights,
            groups=groups,
        )
        runs[name] = {
            "epochs_run": len(history),
            "best_selection_metric": min(
                float(item["selection_metric_value"]) for item in history
            ),
            "metrics": metrics,
            "history": history,
        }
        torch.save(
            {
                "model": {key: value.cpu() for key, value in model.state_dict().items()},
                "model_type": name,
                "seed": args.seed,
                "stage": "M0-development-only",
                "not_for_m1": True,
                "parameter_count": parameter_counts[name],
                "node_features": groups.node_names,
                "edge_features": groups.edge_names,
                "selection_metric": training["selection_metric"],
                "endpoint_shuffle_mode": shuffle_mode,
                "schema_sha256": schema_fingerprint(groups),
                "config_sha256": config_fingerprint(config),
                "preprocessing_code_sha256": preprocessing_code_fingerprint(),
                "training_code_sha256": training_code_fingerprint(),
            },
            checkpoint_paths[name],
        )

    family_comparison: dict[str, Any] = {}
    for family in FAMILY_KEYS:
        values = {
            name: float(run["metrics"]["aggregate"][f"{family}_mse_mean"])
            for name, run in runs.items()
        }
        graph = values["graph"]
        family_comparison[family] = {
            **values,
            "constant": float(constant["aggregate"][f"{family}_mse_mean"]),
            "graph_gain_over_local": (values["local"] - graph) / values["local"],
            "graph_gain_over_set_context": (
                values["set_context"] - graph
            ) / values["set_context"],
            "graph_gain_over_shuffled_graph": (
                values["shuffled_graph"] - graph
            ) / values["shuffled_graph"],
        }

    report = {
        "status": "development_only_not_an_admissibility_audit",
        "warning": (
            "Select-normal was used for early stopping. These metrics diagnose "
            "optimization and masking only and cannot admit a family to M1."
        ),
        "loaded_artifacts": ["train_normal.pt", "select_normal.pt"],
        "seed": args.seed,
        "selection_metric": training["selection_metric"],
        "development_report_mask_seeds": list(DEVELOPMENT_REPORT_MASK_SEEDS),
        "parameter_counts": parameter_counts,
        "endpoint_shuffle_mode": shuffle_mode,
        "constant": constant,
        "runs": runs,
        "family_comparison": family_comparison,
        "config_sha256": config_fingerprint(config),
        "preprocessing_code_sha256": preprocessing_code_fingerprint(),
        "training_code_sha256": training_code_fingerprint(),
        "orchestration_sha256": _sha256(Path(__file__)),
        "schema_sha256": schema_fingerprint(groups),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "families": family_comparison}, indent=2))


if __name__ == "__main__":
    main()
