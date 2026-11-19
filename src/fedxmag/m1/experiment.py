"""Pre-registered M1 experiment phases.

``register`` (requires a passed independent M0 confirmation) ->
``train(arm, seed)`` (fit + selection normal only) ->
``geometry`` (normal-only diagnostics; freezes every detector and threshold) ->
``test`` (opens test-normal and test-attack exactly once).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn

from fedxmag.config import config_fingerprint, feature_groups_from_config, load_stage1_config
from fedxmag.confirmation import (
    masker_from_config,
    model_builders,
    runtime_metadata,
    sha256_file,
)
from fedxmag.losses import FAMILY_KEYS
from fedxmag.m1.covariance import linear_cka
from fedxmag.m1.detection import (
    MahalanobisDetector,
    cluster_bootstrap_detection,
    conformal_threshold,
    descriptor_matrix,
    detection_summary,
    spearman,
)
from fedxmag.m1.objective import NormalSupportSettings
from fedxmag.m1.training import SELECTION_MODES, embed_clean, train_m1
from fedxmag.models import M0GraphModel
from fedxmag.schema import FeatureGroups, GraphSample
from fedxmag.serialization import load_graph_samples

CODE_ROOT = Path(__file__).resolve().parents[3]
CODE_FILES = (
    "src/fedxmag/confirmation.py",
    "src/fedxmag/m1/__init__.py",
    "src/fedxmag/m1/covariance.py",
    "src/fedxmag/m1/detection.py",
    "src/fedxmag/m1/experiment.py",
    "src/fedxmag/m1/objective.py",
    "src/fedxmag/m1/training.py",
    "scripts/run_m1.py",
)
NORMAL_ROLES = ("fit", "selection", "calibration")
DETERMINISTIC_MEMBER = "deterministic"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def code_hashes() -> dict[str, str]:
    return {name: sha256_file(CODE_ROOT / name) for name in CODE_FILES}


def _write_new(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite M1 artifact: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@dataclass
class M1Context:
    root: Path
    output_dir: Path
    protocol: dict[str, Any]
    protocol_sha256: str
    config: dict[str, Any]
    groups: FeatureGroups
    settings: dict[str, NormalSupportSettings]

    @property
    def arms(self) -> list[str]:
        return list(self.protocol["arms"])

    @property
    def seeds(self) -> list[int]:
        return [int(seed) for seed in self.protocol["training"]["seeds"]]

    @property
    def prepared_dir(self) -> Path:
        return self.root / self.protocol["data"]["prepared_dir"]

    def checkpoint(self, arm: str, seed: int) -> Path:
        return self.output_dir / f"m1_{arm}_seed_{seed}.pt"

    def history(self, arm: str, seed: int) -> Path:
        return self.output_dir / f"m1_{arm}_seed_{seed}.json"


def load_context(protocol_path: Path, root: Path, output_dir: Path) -> M1Context:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    config = load_stage1_config(root / protocol["base_config"])
    if config_fingerprint(config) != protocol["base_config_sha256"]:
        raise ValueError("The M1 base configuration differs from its registered fingerprint")
    groups = feature_groups_from_config(config)
    defaults = protocol["objective_defaults"]
    settings = {
        arm: NormalSupportSettings(**{**defaults, **(spec.get("overrides") or {})})
        for arm, spec in protocol["arms"].items()
    }
    for arm, spec in protocol["arms"].items():
        if spec["selection"] not in SELECTION_MODES:
            raise ValueError(f"Arm {arm} has an unknown selection mode")
    if protocol["protocol"]["primary_arm"] not in settings:
        raise ValueError("The primary arm must be one of the registered arms")
    roles = protocol["data"]["roles"]
    if set(protocol["data"]["never_opened"]) & set(roles.values()):
        raise ValueError("A registered data role uses a never-opened artifact")
    return M1Context(
        root=root,
        output_dir=output_dir,
        protocol=protocol,
        protocol_sha256=sha256_file(protocol_path),
        config=config,
        groups=groups,
        settings=settings,
    )


def register(context: M1Context) -> dict[str, Any]:
    decision_path = context.root / context.protocol["protocol"]["prerequisite_m0_decision"]
    if not decision_path.exists():
        raise RuntimeError("M1 is blocked: no independent M0 confirmation decision exists")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "pass":
        raise RuntimeError("M1 is blocked: the independent M0 confirmation did not pass")
    hashes: dict[str, str] = {}
    for role, name in context.protocol["data"]["roles"].items():
        observed = sha256_file(context.prepared_dir / name)
        expected = context.protocol["data"]["file_sha256"].get(name)
        if expected is not None and expected != observed:
            raise ValueError(f"Prepared artifact {name} differs from its registered hash")
        hashes[role] = observed
    context.output_dir.mkdir(parents=True, exist_ok=True)
    registration = {
        "status": "registered",
        "registered_at_utc": _now(),
        "protocol_sha256": context.protocol_sha256,
        "code": code_hashes(),
        "m0_decision_sha256": sha256_file(decision_path),
        "m0_decision_status": decision["status"],
        "data_role_sha256": hashes,
    }
    _write_new(context.output_dir / "registration.json", registration)
    return registration


def _registration(context: M1Context) -> dict[str, Any]:
    path = context.output_dir / "registration.json"
    if not path.exists():
        raise FileNotFoundError("The M1 protocol has not been registered")
    registration = json.loads(path.read_text(encoding="utf-8"))
    if registration["protocol_sha256"] != context.protocol_sha256:
        raise ValueError("The M1 protocol changed after registration")
    if registration["code"] != code_hashes():
        raise ValueError("M1 code changed after registration")
    return registration


def _open(context: M1Context, registration: Mapping[str, Any], role: str) -> list[GraphSample]:
    path = context.prepared_dir / context.protocol["data"]["roles"][role]
    if sha256_file(path) != registration["data_role_sha256"][role]:
        raise ValueError(f"{path.name} changed after registration")
    return load_graph_samples(path, context.groups)


def _configure(context: M1Context, threads: int) -> None:
    torch.set_num_threads(threads)
    training = context.config["training"]
    torch.use_deterministic_algorithms(
        bool(training["deterministic_algorithms"]),
        warn_only=bool(training["deterministic_warn_only"]),
    )


def _graph_model(context: M1Context) -> nn.Module:
    model = context.config["model"]
    return M0GraphModel(
        context.groups.node_dim,
        context.groups.edge_dim,
        hidden_dim=int(model["hidden_dim"]),
        graph_dim=int(model["graph_dim"]),
        num_layers=int(model["num_layers"]),
        dropout=float(model["dropout"]),
    )


def train_arm(
    context: M1Context, arm: str, seed: int, threads: int = 1, device: str = "cpu"
) -> dict[str, Any]:
    registration = _registration(context)
    if arm not in context.settings or seed not in context.seeds:
        raise ValueError(f"Arm {arm!r} with seed {seed} is not registered")
    checkpoint, history_path = context.checkpoint(arm, seed), context.history(arm, seed)
    if checkpoint.exists() or history_path.exists():
        raise FileExistsError(f"Refusing to overwrite M1 run {arm}/{seed}")
    _configure(context, threads)
    train_samples = _open(context, registration, "fit")
    select_samples = _open(context, registration, "selection")
    training = context.protocol["training"]
    started = time.perf_counter()
    torch.manual_seed(seed)
    model = _graph_model(context)
    history = train_m1(
        model,
        train_samples,
        select_samples,
        masker_from_config(context.config, context.groups),
        context.settings[arm],
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        validation_batch_size=int(training["validation_batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        validation_mask_seeds=[int(item) for item in training["validation_mask_seeds"]],
        selection_mode=context.protocol["arms"][arm]["selection"],
        loss_weights={key: float(context.config["loss"][key]) for key in FAMILY_KEYS},
        seed=seed,
        groups=context.groups,
        device=device,
    )
    values = [item["selection_metric_value"] for item in history]
    finite = [value if math.isfinite(value) else math.inf for value in values]
    best_epoch = int(np.argmin(finite)) + 1
    summary = {
        "arm": arm,
        "seed": seed,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_selection_metric": finite[best_epoch - 1],
        "wall_seconds": time.perf_counter() - started,
    }
    torch.save(
        {
            "model": {key: value.cpu() for key, value in model.state_dict().items()},
            "protocol_sha256": context.protocol_sha256,
            "code": code_hashes(),
            "runtime": runtime_metadata(device),
            **summary,
        },
        checkpoint,
    )
    _write_new(
        history_path,
        {**summary, "settings": asdict(context.settings[arm]), "history": history},
    )
    return summary


def _load_arm(context: M1Context, arm: str, seed: int) -> nn.Module:
    payload = torch.load(context.checkpoint(arm, seed), map_location="cpu", weights_only=True)
    if payload["protocol_sha256"] != context.protocol_sha256 or payload["code"] != code_hashes():
        raise ValueError(f"M1 checkpoint {arm}/{seed} does not match the registration")
    model = _graph_model(context)
    model.load_state_dict(payload["model"])
    return model.eval()


def _load_m0_confirmed(context: M1Context, predictor: str, seed: int) -> nn.Module:
    path = context.root / context.protocol["m0_confirmation_dir"] / f"m0_{predictor}_seed_{seed}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = model_builders(context.config, context.groups)[predictor]()
    model.load_state_dict(payload["model"])
    return model.eval()


@dataclass
class Method:
    name: str
    family: str
    estimator: str
    standardize: bool
    members: dict[str, Callable[[Sequence[GraphSample]], Tensor]]


def _methods(context: M1Context) -> list[Method]:
    batch = int(context.protocol["training"]["validation_batch_size"])
    score_estimator = context.protocol["scoring"]["score_time_estimator"]
    groups = context.groups

    def embedder(model: nn.Module) -> Callable[[Sequence[GraphSample]], Tensor]:
        return lambda samples: embed_clean(model, samples, groups, batch_size=batch)

    methods: list[Method] = []
    for name, spec in context.protocol["baselines"]["descriptor_methods"].items():
        kind = spec["descriptors"]
        methods.append(
            Method(
                name,
                "handcrafted",
                spec["estimator"],
                True,
                {
                    DETERMINISTIC_MEMBER: lambda samples, kind=kind: descriptor_matrix(
                        samples, kind, groups
                    )
                },
            )
        )
    m0_seeds = [int(seed) for seed in context.protocol["m0_confirmation_seeds"]]
    for predictor in context.protocol["baselines"]["m0_confirmed_embeddings"]:
        methods.append(
            Method(
                f"m0_confirmed_{predictor}",
                "m0_confirmed",
                score_estimator,
                False,
                {
                    str(seed): embedder(_load_m0_confirmed(context, predictor, seed))
                    for seed in m0_seeds
                },
            )
        )
    for arm in context.arms:
        methods.append(
            Method(
                arm,
                "m1_arm",
                score_estimator,
                False,
                {str(seed): embedder(_load_arm(context, arm, seed)) for seed in context.seeds},
            )
        )
    return methods


def _energy_summary(scores: np.ndarray, dimension: int) -> dict[str, float]:
    return {
        "mean_over_d": float(np.mean(scores) / dimension),
        "median_over_d": float(np.median(scores) / dimension),
        "q95_over_d": float(np.quantile(scores, 0.95) / dimension),
        "q99_over_d": float(np.quantile(scores, 0.99) / dimension),
    }


def _training_alphas(context: M1Context, arm: str, seed: int) -> dict[str, float] | None:
    record = json.loads(context.history(arm, seed).read_text(encoding="utf-8"))
    best = int(record["best_epoch"])
    alphas = [item["training"].get("alpha") for item in record["history"][:best]]
    alphas = [float(value) for value in alphas if value is not None]
    if not alphas:
        return None
    return {"trajectory_mean": float(np.mean(alphas)), "best_epoch": alphas[-1]}


def _fit_members(
    context: M1Context, methods: Sequence[Method], samples: Mapping[str, Sequence[GraphSample]]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Representations, fitted detectors, and normal-role scores for every member."""

    fitted: dict[str, dict[str, dict[str, Any]]] = {}
    for method in methods:
        fitted[method.name] = {}
        for member, represent in method.members.items():
            representations = {role: represent(values) for role, values in samples.items()}
            detector = MahalanobisDetector.fit(
                representations["fit"], method.estimator, standardize=method.standardize
            )
            fitted[method.name][member] = {
                "representations": representations,
                "detector": detector,
                "scores": {
                    role: detector.score(values).numpy()
                    for role, values in representations.items()
                },
            }
    return fitted


def _replay_detectors(
    context: M1Context, method: Method, member: str, fit_representation: Tensor
) -> dict[str, MahalanobisDetector]:
    """Counterfactual score-time shrinkage replay on one frozen checkpoint."""

    variants = {
        "dynamic_ledoit_wolf": MahalanobisDetector.fit(fit_representation, "ledoit_wolf"),
        "fixed": MahalanobisDetector.fit(
            fit_representation,
            "fixed",
            fixed_alpha=float(context.protocol["scoring"]["replay_fixed_alpha"]),
        ),
        "empirical": MahalanobisDetector.fit(fit_representation, "empirical"),
    }
    if method.family == "m1_arm":
        alphas = _training_alphas(context, method.name, int(member))
        if alphas is not None:
            for key, value in alphas.items():
                variants[f"training_alpha_{key}"] = MahalanobisDetector.fit(
                    fit_representation, "fixed", alpha_override=value
                )
    return variants


def geometry(context: M1Context) -> dict[str, Any]:
    registration = _registration(context)
    report_path = context.output_dir / "geometry_report.json"
    if report_path.exists():
        raise FileExistsError("The geometry phase has already run")
    missing = [
        str(context.checkpoint(arm, seed))
        for arm in context.arms
        for seed in context.seeds
        if not context.checkpoint(arm, seed).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Every registered M1 run must exist first: {missing}")
    samples = {role: _open(context, registration, role) for role in NORMAL_ROLES}
    methods = _methods(context)
    fitted = _fit_members(context, methods, samples)
    targets = [float(item) for item in context.protocol["scoring"]["fpr_targets"]]
    stability = context.protocol["geometry"]
    report_methods: dict[str, Any] = {}
    energies_cache: dict[str, np.ndarray] = {}
    for method in methods:
        members: dict[str, Any] = {}
        for member, item in fitted[method.name].items():
            scores = item["scores"]
            dimension = item["detector"].dimension
            thresholds = {
                f"{target:g}": conformal_threshold(scores["calibration"], 1.0 - target)
                for target in targets
            }
            fit_values = item["representations"]["fit"]
            generator = torch.Generator().manual_seed(int(stability["split_stability_seed"]))
            correlations: list[float] = []
            q99: list[float] = []
            for _ in range(int(stability["split_stability_resamples"])):
                order = torch.randperm(fit_values.shape[0], generator=generator)
                half = fit_values.shape[0] // 2
                halves = [
                    MahalanobisDetector.fit(
                        fit_values[order[:half]], method.estimator, standardize=method.standardize
                    ),
                    MahalanobisDetector.fit(
                        fit_values[order[half:]], method.estimator, standardize=method.standardize
                    ),
                ]
                calibration_values = item["representations"]["calibration"]
                split_scores = [
                    detector.score(calibration_values).numpy() for detector in halves
                ]
                correlations.append(spearman(*split_scores))
                q99.extend(
                    float(np.quantile(values, 0.99) / detector.dimension)
                    for values, detector in zip(split_scores, halves, strict=True)
                )
            replay = {
                name: {
                    "alpha": float(detector.estimate.alpha),
                    "shape_condition_number": detector.diagnostics()["shape_condition_number"],
                    "calibration_thresholds": {
                        f"{target:g}": conformal_threshold(
                            detector.score(item["representations"]["calibration"]).numpy(),
                            1.0 - target,
                        )
                        for target in targets
                    },
                    "selection_fpr_at_calibrated": {
                        f"{target:g}": float(
                            np.mean(
                                detector.score(item["representations"]["selection"]).numpy()
                                > conformal_threshold(
                                    detector.score(item["representations"]["calibration"]).numpy(),
                                    1.0 - target,
                                )
                            )
                        )
                        for target in targets
                    },
                }
                for name, detector in _replay_detectors(
                    context, method, member, fit_values
                ).items()
            }
            members[member] = {
                "representation_dim": dimension,
                "covariance": item["detector"].diagnostics(),
                "energy": {
                    role: _energy_summary(values, dimension) for role, values in scores.items()
                },
                "calibration_thresholds": thresholds,
                "selection_fpr_at_calibrated": {
                    key: float(np.mean(scores["selection"] > value))
                    for key, value in thresholds.items()
                },
                "fit_fpr_at_calibrated": {
                    key: float(np.mean(scores["fit"] > value)) for key, value in thresholds.items()
                },
                "heldout_to_fit_mean_energy_ratio": float(
                    np.mean(scores["calibration"]) / np.mean(scores["fit"])
                ),
                "split_stability": {
                    "mean_spearman": float(np.mean(correlations)),
                    "min_spearman": float(np.min(correlations)),
                    "q99_over_d_mean": float(np.mean(q99)),
                    "q99_over_d_cv": float(np.std(q99) / np.mean(q99)),
                },
                "shrinkage_replay": replay,
            }
            for role in ("selection", "calibration"):
                energies_cache[f"{method.name}#{member}#{role}"] = scores[role] / dimension
        entry: dict[str, Any] = {
            "family": method.family,
            "estimator": method.estimator,
            "members": members,
        }
        member_names = list(fitted[method.name])
        if len(member_names) > 1:
            entry["seed_stability"] = {
                "pairwise_selection_cka_mean": float(
                    np.mean(
                        [
                            linear_cka(
                                fitted[method.name][left]["representations"]["selection"],
                                fitted[method.name][right]["representations"]["selection"],
                            )
                            for left, right in combinations(member_names, 2)
                        ]
                    )
                ),
                **{
                    f"{key}_std_across_seeds": float(
                        np.std([members[name]["covariance"][key] for name in member_names])
                    )
                    for key in (
                        "raw_condition_number",
                        "raw_effective_rank_entropy",
                        "shape_condition_number",
                        "alpha",
                    )
                },
                "calibration_q99_over_d_std_across_seeds": float(
                    np.std([members[name]["energy"]["calibration"]["q99_over_d"]
                            for name in member_names])
                ),
            }
        report_methods[method.name] = entry

    representation_change: dict[str, Any] = {}
    for arm in context.arms:
        if arm == "m0_recon":
            continue
        representation_change[arm] = {
            str(seed): linear_cka(
                fitted[arm][str(seed)]["representations"]["selection"],
                fitted["m0_recon"][str(seed)]["representations"]["selection"],
            )
            for seed in context.seeds
        }
    if "m0_confirmed_graph" in fitted:
        representation_change["m0_recon_vs_m0_confirmed_graph"] = {
            str(seed): linear_cka(
                fitted["m0_recon"][str(seed)]["representations"]["selection"],
                fitted["m0_confirmed_graph"][str(seed)]["representations"]["selection"],
            )
            for seed in context.seeds
            if str(seed) in fitted["m0_confirmed_graph"]
        }

    reconstruction = {}
    for arm in context.arms:
        reconstruction[arm] = {}
        for seed in context.seeds:
            record = json.loads(context.history(arm, seed).read_text(encoding="utf-8"))
            validation = record["history"][int(record["best_epoch"]) - 1]["validation"]
            reconstruction[arm][str(seed)] = {
                "best_epoch": record["best_epoch"],
                "epochs_run": record["epochs_run"],
                **{f"{key}_mse": validation[f"{key}_mse"] for key in FAMILY_KEYS},
                "worst_family_mse": validation["worst_family_mse"],
            }

    report = {
        "phase": "geometry",
        "status": "complete",
        "loaded_roles": list(NORMAL_ROLES),
        "protocol_sha256": context.protocol_sha256,
        "code": code_hashes(),
        "counts": {role: len(values) for role, values in samples.items()},
        "methods": report_methods,
        "representation_change_cka_vs_m0_recon": representation_change,
        "selection_reconstruction_at_best_epoch": reconstruction,
        "finished_at_utc": _now(),
    }
    np.savez_compressed(context.output_dir / "geometry_energies.npz", **energies_cache)
    _write_new(report_path, report)
    return report


def _category_recall(
    attack_scores: np.ndarray, attack_samples: Sequence[GraphSample], threshold: float
) -> dict[str, dict[str, float]]:
    categories: dict[str, list[int]] = {}
    for index, sample in enumerate(attack_samples):
        for category in sample.metadata.get("attack_categories", []):
            categories.setdefault(str(category), []).append(index)
    return {
        category: {
            "windows": len(indices),
            "tpr": float(np.mean(attack_scores[indices] > threshold)),
        }
        for category, indices in sorted(categories.items())
    }


def sealed_test(context: M1Context) -> dict[str, Any]:
    registration = _registration(context)
    geometry_path = context.output_dir / "geometry_report.json"
    report_path = context.output_dir / "test_report.json"
    if not geometry_path.exists():
        raise FileNotFoundError("The normal-only geometry phase must run before the test")
    if report_path.exists():
        raise FileExistsError("The sealed test partition has already been evaluated")
    frozen = json.loads(geometry_path.read_text(encoding="utf-8"))
    normal_samples = {role: _open(context, registration, role) for role in ("fit", "calibration")}
    methods = _methods(context)
    fitted = _fit_members(context, methods, normal_samples)
    targets = [float(item) for item in context.protocol["scoring"]["fpr_targets"]]
    for method in methods:
        for member, item in fitted[method.name].items():
            stored = frozen["methods"][method.name]["members"][member]["calibration_thresholds"]
            for target in targets:
                recomputed = conformal_threshold(item["scores"]["calibration"], 1.0 - target)
                if not math.isclose(recomputed, stored[f"{target:g}"], rel_tol=1e-9):
                    raise AssertionError(f"{method.name}/{member} threshold is not frozen")

    opened_at = _now()
    test_normal = _open(context, registration, "test_normal")
    test_attack = _open(context, registration, "test_attack")
    category_target = float(context.protocol["test"]["per_category_fpr_target"])
    report_methods: dict[str, Any] = {}
    bootstrap_scores: dict[str, dict[str, np.ndarray]] = {}
    members_by_method: dict[str, list[str]] = {}
    score_cache: dict[str, np.ndarray] = {}
    for method in methods:
        members: dict[str, Any] = {}
        for member, item in fitted[method.name].items():
            represent = method.members[member]
            detector = item["detector"]
            normal_representation = represent(test_normal)
            attack_representation = represent(test_attack)
            normal_scores = detector.score(normal_representation).numpy()
            attack_scores = detector.score(attack_representation).numpy()
            calibration_scores = item["scores"]["calibration"]
            summary = detection_summary(calibration_scores, normal_scores, attack_scores, targets)
            summary["per_category_tpr"] = _category_recall(
                attack_scores,
                test_attack,
                conformal_threshold(calibration_scores, 1.0 - category_target),
            )
            if method.family != "handcrafted":
                replay: dict[str, Any] = {}
                for name, variant in _replay_detectors(
                    context, method, member, item["representations"]["fit"]
                ).items():
                    calibration_variant = variant.score(
                        item["representations"]["calibration"]
                    ).numpy()
                    normal_variant = variant.score(normal_representation).numpy()
                    attack_variant = variant.score(attack_representation).numpy()
                    replay[name] = {
                        "alpha": float(variant.estimate.alpha),
                        **detection_summary(
                            calibration_variant, normal_variant, attack_variant, targets
                        ),
                    }
                summary["shrinkage_replay"] = replay
            members[member] = summary
            key = f"{method.name}#{member}"
            bootstrap_scores[key] = {
                "calibration": calibration_scores,
                "normal": normal_scores,
                "attack": attack_scores,
            }
            members_by_method.setdefault(method.name, []).append(key)
            for role, values in bootstrap_scores[key].items():
                score_cache[f"{key}#{role}"] = values
        aggregate: dict[str, Any] = {}
        for metric in ("auroc", "auprc"):
            values = [members[name][metric] for name in members]
            aggregate[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for target in targets:
            for metric in ("tpr", "fpr", "f1", "precision"):
                values = [members[name]["at_calibrated_fpr"][f"{target:g}"][metric]
                          for name in members]
                aggregate[f"{metric}@{target:g}"] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }
        report_methods[method.name] = {
            "family": method.family,
            "estimator": method.estimator,
            "aggregate_over_members": aggregate,
            "members": members,
        }

    clusters = {
        role: [str(sample.metadata["session_key"]) for sample in values]
        for role, values in (
            ("calibration", normal_samples["calibration"]),
            ("normal", test_normal),
            ("attack", test_attack),
        )
    }
    pairs = [
        (left, right)
        for left, right in context.protocol["test"]["primary_comparisons"]
        if left in members_by_method and right in members_by_method
    ]
    primary = context.protocol["protocol"]["primary_arm"]
    pairs.extend(
        (primary, name)
        for name in members_by_method
        if name != primary and (primary, name) not in pairs
    )
    bootstrap = cluster_bootstrap_detection(
        bootstrap_scores,
        members_by_method,
        clusters,
        pairs,
        targets,
        int(context.protocol["test"]["bootstrap_resamples"]),
        int(context.protocol["test"]["bootstrap_seed"]),
    )
    np.savez_compressed(context.output_dir / "test_scores.npz", **score_cache)
    report = {
        "phase": "test",
        "protocol_sha256": context.protocol_sha256,
        "code": code_hashes(),
        "geometry_report_sha256": sha256_file(geometry_path),
        "test_opened_at_utc": opened_at,
        "counts": {
            "test_normal": len(test_normal),
            "test_attack": len(test_attack),
            "calibration": len(normal_samples["calibration"]),
        },
        "methods": report_methods,
        "paired_cluster_bootstrap": bootstrap,
        "threshold_rule": (
            "split-conformal on calibration-normal; attack data never set a threshold"
        ),
        "finished_at_utc": _now(),
    }
    _write_new(report_path, report)
    return report


__all__ = [
    "M1Context",
    "code_hashes",
    "geometry",
    "load_context",
    "register",
    "sealed_test",
    "train_arm",
]
