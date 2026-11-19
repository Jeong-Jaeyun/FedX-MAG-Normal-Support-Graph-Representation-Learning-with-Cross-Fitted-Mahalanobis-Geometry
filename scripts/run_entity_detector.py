"""Build, use, and replay the preferred frozen-M0 node detector.

The existing confirmation protocol identifies immutable inputs. Build opens
only train-normal and calibration-normal. Replay checks already-consumed data
against archived scores; it is a software check, not a new confirmation.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fedxmag.config import (
    config_fingerprint,
    feature_groups_from_config,
    load_stage1_config,
    preprocessing_code_fingerprint,
    training_code_fingerprint,
)
from fedxmag.confirmation import sha256_file
from fedxmag.m1.detection import detection_summary
from fedxmag.m1.entity import StateGraphModel
from fedxmag.scoring import FPR_TARGETS, METHOD, FrozenNodeDetector
from fedxmag.serialization import load_graph_samples

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/entity_scoring_confirmation_protocol.yaml"
OUTPUT = ROOT / "output/entity_detector"
PRIMARY = "m0_confirmed_graph|node|top3_mean"


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def verified_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    registration = json.loads((ROOT / "output/entity_confirmation/registration.json").read_text())
    if sha256_file(PROTOCOL) != registration["protocol_sha256"]:
        raise ValueError("The reference confirmation protocol changed")
    for name, expected in registration["code"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Frozen confirmation code changed: {name}")
    if protocol["protocol"]["primary"] != PRIMARY:
        raise ValueError("Unexpected reference scoring method")
    return protocol, registration


def data_path(protocol: dict[str, Any], registration: dict[str, Any], role: str) -> Path:
    reference = protocol["data"]["roles"][role]
    prefix, name = reference.split("/", 1)
    directory = "prepared_dir" if prefix == "prepared" else "audit_band_dir"
    path = ROOT / protocol["data"][directory] / name
    if sha256_file(path) != registration["data"][reference]:
        raise ValueError(f"Data differ from the confirmation registration: {reference}")
    return path


def build(output: Path) -> None:
    protocol, registration = verified_inputs()
    spec = protocol["encoders"]["m0_confirmed_graph"]
    destinations = [output / f"detector_seed_{seed}.pt" for seed in spec["seeds"]]
    if any(path.exists() for path in (*destinations, output / "build_report.json")):
        raise FileExistsError("Detector output already exists; choose a new --output-dir")
    base_path = ROOT / protocol["base_config"]
    config = load_stage1_config(base_path)
    groups = feature_groups_from_config(config)
    m0_dir = ROOT / "output/m0/xiiotid_stage1_v3b_confirmation"
    decision_path = m0_dir / "confirmation_decision.json"
    if json.loads(decision_path.read_text())["status"] != "pass":
        raise ValueError("The independent M0 confirmation must pass")
    m0_registration = json.loads((m0_dir / "registration.json").read_text())
    if config_fingerprint(config) != m0_registration["frozen_inputs"]["base_config_sha256"]:
        raise ValueError("The encoder base configuration changed")
    frozen = m0_registration["frozen_inputs"]
    if (
        preprocessing_code_fingerprint(ROOT) != frozen["preprocessing_code_sha256"]
        or training_code_fingerprint(ROOT) != frozen["training_code_sha256"]
    ):
        raise ValueError("Frozen preprocessing or M0 training code changed")
    train_path = data_path(protocol, registration, "fit")
    calibration_path = data_path(protocol, registration, "calibration")
    if train_path.resolve() == calibration_path.resolve():
        raise ValueError("Training and calibration roles must be separate")
    train = load_graph_samples(train_path, groups)
    calibration = load_graph_samples(calibration_path, groups)
    prepared = ROOT / protocol["data"]["prepared_dir"]
    standardizer = prepared / "standardizer.pt"
    if sha256_file(standardizer) != m0_registration["data_file_sha256"]["standardizer.pt"]:
        raise ValueError("The frozen preprocessing standardizer changed")
    kwargs = {"node_dim": groups.node_dim, "edge_dim": groups.edge_dim, **config["model"]}
    batch = int(protocol["scoring"]["embedding_batch_size"])
    report: dict[str, Any] = {
        "method": METHOD,
        "encoder_training": "frozen_M0_reconstruction_only",
        "normal_support_training_weight": 0.0,
        "primary_fpr_target": 0.05,
        "aggregation": "mean_of_min_3_highest_node_energies",
        "seed_policy": "independent_detectors_and_thresholds; no_seed_ensemble",
        "counts": {"train_normal": len(train), "calibration_normal": len(calibration)},
        "members": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    for seed, destination in zip(spec["seeds"], destinations, strict=True):
        checkpoint = ROOT / spec["pattern"].format(seed=seed)
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != registration["checkpoints"][f"m0_confirmed_graph#{seed}"]:
            raise ValueError(f"Frozen M0 checkpoint changed: {seed}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            payload["stage"] != "M0-confirmation"
            or payload["model_type"] != "graph"
            or payload["seed"] != seed
            or tuple(payload["node_features"]) != groups.node_names
            or tuple(payload["edge_features"]) != groups.edge_names
        ):
            raise ValueError(f"Incompatible M0 checkpoint: {seed}")
        model = StateGraphModel(**kwargs)
        model.load_state_dict(payload["model"])
        provenance = {
            "seed": seed,
            "checkpoint_sha256": checkpoint_hash,
            "base_config_sha256": config_fingerprint(config),
            "reference_protocol_sha256": registration["protocol_sha256"],
            "m0_decision_sha256": sha256_file(decision_path),
            "train_normal_sha256": sha256_file(train_path),
            "calibration_normal_sha256": sha256_file(calibration_path),
            "feature_standardizer_sha256": sha256_file(standardizer),
            "source_code": {
                **registration["code"],
                "src/fedxmag/scoring.py": sha256_file(ROOT / "src/fedxmag/scoring.py"),
                "scripts/run_entity_detector.py": sha256_file(Path(__file__)),
            },
        }
        detector = FrozenNodeDetector.fit(
            model, kwargs, groups, train, calibration, batch_size=batch, provenance=provenance
        )
        detector.save(destination)
        report["members"][str(seed)] = {
            "artifact": destination.name,
            "sha256": sha256_file(destination),
            "thresholds": detector.thresholds,
            "reference_node_count": sum(s.node_features.shape[0] for s in train),
        }
        print(json.dumps({"built_seed": seed, "thresholds": detector.thresholds}), flush=True)
    report["finished_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    write_json(output / "build_report.json", report)


def replay(output: Path) -> None:
    protocol, registration = verified_inputs()
    destination = output / "replay_report.json"
    scores_path = output / "replay_scores.pt"
    if destination.exists() or scores_path.exists():
        raise FileExistsError("Replay output already exists")
    build_report = json.loads((output / "build_report.json").read_text())
    datasets = {
        "test": ("replay_test_normal", "replay_test_attack"),
        "audit": ("confirm_normal", "confirm_attack"),
    }
    report: dict[str, Any] = {
        "status": "replay_of_consumed_data",
        "method": METHOD,
        "seed_policy": build_report["seed_policy"],
        "datasets": {},
    }
    saved_scores: dict[str, torch.Tensor] = {}
    for name, (normal_role, attack_role) in datasets.items():
        archive_dir = ROOT / (
            "output/entity_confirmation" if name == "audit" else "output/diagnosis"
        )
        archive_path = archive_dir / (
            "confirmation_scores.npz" if name == "audit" else "entity_scoring_scores.npz"
        )
        with np.load(archive_path) as archive:
            archived = {
                key: archive[key].copy() for key in archive.files if key.startswith(f"{PRIMARY}#")
            }
        per_seed = {}
        for seed, member in build_report["members"].items():
            artifact = output / member["artifact"]
            if sha256_file(artifact) != member["sha256"]:
                raise ValueError(f"Detector artifact changed: {seed}")
            detector = FrozenNodeDetector.load(artifact)
            normal = load_graph_samples(
                data_path(protocol, registration, normal_role), detector.groups
            )
            attack = load_graph_samples(
                data_path(protocol, registration, attack_role), detector.groups
            )
            n_score, _ = detector.score(normal)
            a_score, _ = detector.score(attack)
            summary = detection_summary(detector.calibration_scores, n_score, a_score, FPR_TARGETS)
            discrepancies = []
            for role, actual in (
                ("calibration", detector.calibration_scores),
                ("normal", n_score),
                ("attack", a_score),
            ):
                key = f"{PRIMARY}#{seed}#{role}"
                saved_scores[f"{name}#{seed}#{role}"] = torch.from_numpy(actual.copy())
                if key in archived:
                    expected = archived[key]
                    if expected.shape != actual.shape:
                        raise ValueError(f"Archived score alignment differs: {key}")
                    discrepancies.append(float(np.max(np.abs(expected - actual))))
            if len(discrepancies) != 3:
                raise ValueError(
                    f"Archived calibration/normal/attack scores missing: {name}/{seed}"
                )
            summary["archive_max_absolute_score_difference"] = max(discrepancies)
            per_seed[seed] = summary
            print(
                json.dumps(
                    {
                        "dataset": name,
                        "seed": seed,
                        "auroc": summary["auroc"],
                        "archive_max_abs_diff": max(discrepancies),
                    }
                ),
                flush=True,
            )
        metrics = {"auroc": float(np.mean([s["auroc"] for s in per_seed.values()]))}
        for target in FPR_TARGETS:
            for metric in ("tpr", "fpr"):
                metrics[f"{metric}@{target:g}"] = float(
                    np.mean(
                        [s["at_calibrated_fpr"][f"{target:g}"][metric] for s in per_seed.values()]
                    )
                )
        report["datasets"][name] = {
            "mean_of_seed_metrics": metrics,
            "per_seed": per_seed,
            "counts": {"normal": len(normal), "attack": len(attack)},
        }
    differences = [
        s["archive_max_absolute_score_difference"]
        for d in report["datasets"].values()
        for s in d["per_seed"].values()
        if s["archive_max_absolute_score_difference"] is not None
    ]
    report["archive_replay_pass"] = bool(differences and max(differences) <= 1e-9)
    report["max_absolute_score_difference"] = max(differences)
    with scores_path.open("xb") as stream:
        torch.save(saved_scores, stream)
    write_json(destination, report)
    if not report["archive_replay_pass"]:
        raise RuntimeError("Reusable detector did not reproduce the archived scores")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "score", "replay"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Independent detector to use for score (fixed default: 17)",
    )
    parser.add_argument("--input", type=Path, help="Graphs using the frozen v3b standardizer")
    parser.add_argument("--predictions", type=Path, help="New JSON file for score output")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase == "build":
        build(args.output_dir)
    elif args.phase == "replay":
        replay(args.output_dir)
    else:
        if args.input is None or args.predictions is None:
            parser.error("score requires --input and --predictions")
        if args.predictions.exists():
            raise FileExistsError(f"Predictions already exist: {args.predictions}")
        artifact = args.output_dir / f"detector_seed_{args.seed}.pt"
        detector = FrozenNodeDetector.load(artifact)
        samples = load_graph_samples(args.input, detector.groups)
        result = {
            "method": METHOD,
            "seed": detector.provenance["seed"],
            "detector_sha256": sha256_file(artifact),
            "input_sha256": sha256_file(args.input),
            "primary_fpr_target": 0.05,
            "thresholds": detector.thresholds,
            "windows": detector.predict(samples),
        }
        write_json(args.predictions, result)
        print(json.dumps({"windows": len(samples), "predictions": str(args.predictions)}))


if __name__ == "__main__":
    main()
