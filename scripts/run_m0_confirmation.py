"""Pre-registered, independent v3b M0 confirmation.

Three separately invoked phases keep the confirmation partition sealed:

* ``register`` binds the protocol, code, configuration, and data hashes and runs
  a model-free support check on the confirmation partition;
* ``train --seed S`` fits every predictor for one model seed using only
  train-normal (fit) and select-normal (early stopping); and
* ``evaluate`` runs once, after every checkpoint exists, and is the only phase
  that computes outcomes on the confirmation partition. It persists the frozen
  decision before any descriptive analysis is computed.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fedxmag.confirmation import (
    PREDICTORS,
    evaluate_confirmation,
    load_protocol,
    masker_from_config,
    model_builders,
    parameter_count,
    runtime_metadata,
    sha256_file,
    support_check,
    train_predictors_for_seed,
)
from fedxmag.losses import FAMILY_KEYS
from fedxmag.serialization import load_graph_samples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "m0_v3b_confirmation_protocol.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "output" / "m0" / "xiiotid_stage1_v3b_confirmation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("register", "train", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _code_hashes() -> dict[str, str]:
    return {
        "orchestration_sha256": sha256_file(Path(__file__)),
        "confirmation_module_sha256": sha256_file(
            PROJECT_ROOT / "src" / "fedxmag" / "confirmation.py"
        ),
    }


def _refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite confirmation artifacts: {existing}")


def _configure(config: dict[str, Any], threads: int) -> None:
    torch.set_num_threads(threads)
    training = config["training"]
    torch.use_deterministic_algorithms(
        bool(training["deterministic_algorithms"]),
        warn_only=bool(training["deterministic_warn_only"]),
    )
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = bool(training["deterministic_algorithms"])
        torch.backends.cudnn.benchmark = False


def _registration(args: argparse.Namespace, protocol_sha: str) -> dict[str, Any]:
    path = args.output_dir / "registration.json"
    if not path.exists():
        raise FileNotFoundError("The protocol has not been registered")
    registration = json.loads(path.read_text(encoding="utf-8"))
    if registration["status"] != "pass":
        raise RuntimeError("Registration support check did not pass")
    if registration["protocol_sha256"] != protocol_sha:
        raise ValueError("The protocol file changed after registration")
    if registration["code"] != _code_hashes():
        raise ValueError("Confirmation code changed after registration")
    return registration


def register(args: argparse.Namespace) -> None:
    bound = load_protocol(args.protocol, PROJECT_ROOT)
    protocol, config, groups = bound["protocol"], bound["config"], bound["groups"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "registration.json"
    _refuse_existing([path])
    prepared = PROJECT_ROOT / protocol["data"]["prepared_dir"]
    confirmation = load_graph_samples(prepared / protocol["data"]["roles"]["confirmation"], groups)
    support = support_check(
        confirmation,
        masker_from_config(config, groups),
        protocol["confirmation_mask_seeds"],
        groups,
        int(config["training"]["batch_size"]),
        protocol["support_requirements"],
    )
    registration = {
        "status": support["status"],
        "registered_at_utc": _now(),
        "protocol_path": args.protocol.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "protocol_sha256": sha256_file(args.protocol),
        "code": _code_hashes(),
        "frozen_inputs": protocol["frozen_inputs"],
        "data_file_sha256": protocol["data"]["file_sha256"],
        "support_check": support,
        "note": (
            "Support statistics are model-free and label-free; no reconstruction "
            "outcome on the confirmation partition has been computed."
        ),
    }
    path.write_text(json.dumps(registration, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: registration[key] for key in ("status", "protocol_sha256")}))
    if support["status"] != "pass":
        raise RuntimeError(f"Confirmation support check blocked: {support['blockers']}")


def train(args: argparse.Namespace) -> None:
    bound = load_protocol(args.protocol, PROJECT_ROOT)
    protocol, config, groups = bound["protocol"], bound["config"], bound["groups"]
    protocol_sha = sha256_file(args.protocol)
    _registration(args, protocol_sha)
    if args.seed not in protocol["model_seeds"]:
        raise ValueError(f"Seed {args.seed} is not a registered model seed")
    _configure(config, args.threads)
    runtime = runtime_metadata(args.device)
    report_path = args.output_dir / f"train_seed_{args.seed}.json"
    checkpoint_paths = {
        name: args.output_dir / f"m0_{name}_seed_{args.seed}.pt" for name in PREDICTORS
    }
    _refuse_existing([report_path, *checkpoint_paths.values()])

    prepared = PROJECT_ROOT / protocol["data"]["prepared_dir"]
    roles = protocol["data"]["roles"]
    # Only the fit and early-stopping roles are opened while models are trained.
    train_samples = load_graph_samples(prepared / roles["fit"], groups)
    select_samples = load_graph_samples(prepared / roles["early_stopping"], groups)
    builders = model_builders(config, groups)
    counts = {name: parameter_count(builder()) for name, builder in builders.items()}
    if counts["set_context"] != counts["graph"] or counts["shuffled_graph"] != counts["graph"]:
        raise AssertionError("Capacity-matched controls are not exactly matched")

    def save(name: str, model: torch.nn.Module, run: dict[str, Any]) -> None:
        torch.save(
            {
                "model": {key: value.cpu() for key, value in model.state_dict().items()},
                "model_type": name,
                "seed": args.seed,
                "stage": "M0-confirmation",
                "parameter_count": counts[name],
                "node_features": list(groups.node_names),
                "edge_features": list(groups.edge_names),
                "selection_metric": config["training"]["selection_metric"],
                "protocol_sha256": protocol_sha,
                "frozen_inputs": protocol["frozen_inputs"],
                "code": _code_hashes(),
                "runtime": runtime,
                "best_epoch": run["best_epoch"],
                "epochs_run": run["epochs_run"],
            },
            checkpoint_paths[name],
        )
        summary = {
            key: run[key]
            for key in ("epochs_run", "best_epoch", "best_select_metric", "wall_seconds")
        }
        print(json.dumps({"seed": args.seed, "model": name, **summary}), flush=True)

    _, runs = train_predictors_for_seed(
        builders,
        train_samples,
        select_samples,
        masker_from_config(config, groups),
        config,
        args.seed,
        groups,
        device=args.device,
        on_model_done=save,
    )
    report = {
        "phase": "train",
        "seed": args.seed,
        "loaded_artifacts": [roles["fit"], roles["early_stopping"]],
        "parameter_counts": counts,
        "protocol_sha256": protocol_sha,
        "code": _code_hashes(),
        "runtime": runtime,
        "finished_at_utc": _now(),
        "runs": runs,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _load_checkpoint(
    path: Path, builder: Any, protocol_sha: str, protocol: dict[str, Any]
) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["protocol_sha256"] != protocol_sha:
        raise ValueError(f"{path.name} was trained under a different protocol")
    if payload["frozen_inputs"] != protocol["frozen_inputs"]:
        raise ValueError(f"{path.name} has mismatched frozen fingerprints")
    if payload["code"] != _code_hashes():
        raise ValueError(f"{path.name} was trained by different confirmation code")
    model = builder()
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def evaluate(args: argparse.Namespace) -> None:
    bound = load_protocol(args.protocol, PROJECT_ROOT)
    protocol, config, groups = bound["protocol"], bound["config"], bound["groups"]
    protocol_sha = sha256_file(args.protocol)
    registration = _registration(args, protocol_sha)
    _configure(config, args.threads)
    decision_path = args.output_dir / "confirmation_decision.json"
    report_path = args.output_dir / "confirmation_report.json"
    errors_path = args.output_dir / "confirmation_per_graph_errors.npz"
    _refuse_existing([decision_path, report_path, errors_path])
    seeds = [int(seed) for seed in protocol["model_seeds"]]
    builders = model_builders(config, groups)
    checkpoint_paths = {
        (name, seed): args.output_dir / f"m0_{name}_seed_{seed}.pt"
        for name in PREDICTORS
        for seed in seeds
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Every registered checkpoint must exist first: {missing}")
    training_reports = {
        seed: json.loads(
            (args.output_dir / f"train_seed_{seed}.json").read_text(encoding="utf-8")
        )
        for seed in seeds
    }
    models = {
        key: _load_checkpoint(path, builders[key[0]], protocol_sha, protocol)
        for key, path in checkpoint_paths.items()
    }

    prepared = PROJECT_ROOT / protocol["data"]["prepared_dir"]
    opened_at = _now()
    # The single opening of the confirmation partition for outcomes.
    confirmation = load_graph_samples(prepared / protocol["data"]["roles"]["confirmation"], groups)

    def persist_decision(decision: dict[str, Any]) -> None:
        decision_path.write_text(
            json.dumps(
                {
                    **decision,
                    "protocol_sha256": protocol_sha,
                    "confirmation_opened_at_utc": opened_at,
                    "decided_at_utc": _now(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    result = evaluate_confirmation(
        models,
        confirmation,
        masker_from_config(config, groups),
        seeds,
        [int(seed) for seed in protocol["confirmation_mask_seeds"]],
        protocol["gates"],
        protocol["secondary_descriptive_analyses"],
        int(config["training"]["batch_size"]),
        groups,
        loss_weights={key: float(config["loss"][key]) for key in FAMILY_KEYS},
        device=args.device,
        on_decision=persist_decision,
    )
    np.savez_compressed(
        errors_path,
        constant=result["constant_errors"],
        sessions=np.asarray(result["clusters"]),
        families=np.asarray(FAMILY_KEYS),
        seeds=np.asarray(seeds),
        mask_seeds=np.asarray(protocol["confirmation_mask_seeds"]),
        **result["errors"],
    )
    report = {
        "status": result["decision"]["status"],
        "decision": result["decision"],
        "protocol_sha256": protocol_sha,
        "registration_registered_at_utc": registration["registered_at_utc"],
        "confirmation_opened_at_utc": opened_at,
        "loaded_artifacts": [protocol["data"]["roles"]["confirmation"]],
        "confirmation_normal_graphs": len(confirmation),
        "confirmation_sessions": len(set(result["clusters"])),
        "model_seeds": seeds,
        "confirmation_mask_seeds": protocol["confirmation_mask_seeds"],
        "parameter_counts": training_reports[seeds[0]]["parameter_counts"],
        "training_summary": {
            str(seed): {
                name: {
                    key: training_reports[seed]["runs"][name][key]
                    for key in ("epochs_run", "best_epoch", "best_select_metric", "wall_seconds")
                }
                for name in PREDICTORS
            }
            for seed in seeds
        },
        "training_runtime": {
            str(seed): training_reports[seed]["runtime"] for seed in seeds
        },
        "constant": result["constant"],
        "per_seed_runs": result["per_seed_runs"],
        "secondary_descriptive": result["secondary_descriptive"],
        "code": _code_hashes(),
        "frozen_inputs": protocol["frozen_inputs"],
        "evaluation_runtime": runtime_metadata(args.device),
        "finished_at_utc": _now(),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        key: {
            name: round(value, 4) if isinstance(value, float) else value
            for name, value in entry.items()
        }
        for key, entry in result["decision"]["families"].items()
    }
    print(json.dumps({"status": result["decision"]["status"], "families": summary}, indent=2))


def main() -> None:
    args = parse_args()
    if args.phase == "register":
        register(args)
    elif args.phase == "train":
        if args.seed is None:
            raise ValueError("train requires --seed")
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
