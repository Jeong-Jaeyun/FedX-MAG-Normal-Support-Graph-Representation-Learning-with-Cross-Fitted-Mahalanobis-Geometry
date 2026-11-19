"""Single-source configuration loading and semantic validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from fedxmag.schema import FeatureGroups, feature_groups_for_profile


def feature_groups_from_config(config: dict[str, Any]) -> FeatureGroups:
    features = config["features"]
    profile = str(features.get("timing_profile", "legacy_record_iat_v2"))
    groups = feature_groups_for_profile(profile)
    configured_node = tuple(features["node"]["timing"]) + tuple(
        features["node"]["distribution"]
    )
    configured_edge = tuple(features["edge"]["timing"]) + tuple(
        features["edge"]["distribution"]
    )
    if configured_node != groups.node_names:
        raise ValueError(
            f"Configured node feature order differs from the {profile} code schema"
        )
    if configured_edge != groups.edge_names:
        raise ValueError(
            f"Configured edge feature order differs from the {profile} code schema"
        )
    return groups


def load_stage1_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")

    required = {
        "experiment",
        "data",
        "features",
        "masking",
        "model",
        "controls",
        "loss",
        "training",
        "audit",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Configuration is missing sections: {sorted(missing)}")

    experiment = config["experiment"]
    if experiment.get("stage") != "M0" or int(experiment.get("temporal_length", 0)) != 1:
        raise ValueError("This implementation is frozen to static M0 with temporal_length=1")
    if experiment.get("score_regime") != "closed_window":
        raise ValueError("M0 requires the closed_window score regime")

    data = config["data"]
    if data.get("dataset") != "xiiotid":
        raise ValueError("The first implementation is frozen to X-IIoTID")
    if data.get("split_before_windowing") is not True:
        raise ValueError("split_before_windowing must remain true")
    if not isinstance(data.get("deduplicate_exact_rows"), bool):
        raise ValueError("data.deduplicate_exact_rows must be boolean")
    if data.get("session_split_strategy") not in {
        "exclusive_if_possible",
        "within_session_chronological",
    }:
        raise ValueError("Unknown data.session_split_strategy")
    if data.get("attack_use") != "final_test_only":
        raise ValueError("Attack samples must remain final-test-only")
    if int(data["window_seconds"]) < 1 or int(data["window_stride_seconds"]) < 1:
        raise ValueError("Window size and stride must be positive")

    groups = feature_groups_from_config(config)
    profile = str(config["features"].get("timing_profile", "legacy_record_iat_v2"))
    if profile in {
        "distinct_second_multiplicity_v3",
        "distinct_second_collision_v3",
    }:
        resolution = config["features"].get("timestamp_resolution_seconds")
        if not isinstance(resolution, int) or resolution < 1:
            raise ValueError(
                "features.timestamp_resolution_seconds must be a positive integer"
            )
    elif "timestamp_resolution_seconds" in config["features"]:
        raise ValueError(
            "features.timestamp_resolution_seconds is only valid for the v3 timing profile"
        )
    if groups.node_dim < 1 or groups.edge_dim < 1:
        raise ValueError("Feature schemas must be nonempty")

    fractions = data["split_fractions"]
    expected_splits = {"train", "select", "audit", "calibration", "test"}
    if set(fractions) != expected_splits:
        raise ValueError(f"split_fractions must contain exactly {sorted(expected_splits)}")
    if any(float(value) <= 0 for value in fractions.values()):
        raise ValueError("All split fractions must be positive")
    if abs(sum(float(value) for value in fractions.values()) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to one")

    masking = config["masking"]
    for key in ("attribute_budget", "family_mask_probability"):
        if not 0.0 <= float(masking[key]) <= 1.0:
            raise ValueError(f"masking.{key} must be in [0, 1]")
    if int(masking["minimum_visible_per_entity"]) < 0:
        raise ValueError("minimum_visible_per_entity must be nonnegative")
    if int(masking["minimum_visible_per_graph"]) < 1:
        raise ValueError("minimum_visible_per_graph must be at least one")

    controls = config["controls"]
    if not isinstance(controls.get("adjacency_shuffle_seed"), int):
        raise ValueError("controls.adjacency_shuffle_seed must be an integer")
    shuffle_mode = controls.get("endpoint_shuffle_mode", "destination_only")
    if shuffle_mode not in {"destination_only", "both_endpoints"}:
        raise ValueError(
            "controls.endpoint_shuffle_mode must be destination_only or both_endpoints"
        )

    expected_loss = {
        "node_timing", "node_distribution", "edge_timing", "edge_distribution"
    }
    if set(config["loss"]) != expected_loss:
        raise ValueError(f"loss must contain exactly {sorted(expected_loss)}")
    if any(float(value) <= 0 for value in config["loss"].values()):
        raise ValueError("All loss weights must be positive")

    seeds = config["experiment"].get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("experiment.seeds must contain at least two distinct seeds")
    mask_seeds = config["audit"].get("mask_seeds")
    if not isinstance(mask_seeds, list) or len(mask_seeds) < 2:
        raise ValueError("audit.mask_seeds must contain at least two seeds")
    validation_seeds = config["training"].get("validation_mask_seeds")
    if not isinstance(validation_seeds, list) or len(validation_seeds) < 2:
        raise ValueError("training.validation_mask_seeds must contain at least two seeds")
    if config["training"].get("selection_metric") not in {
        "validation_mask_mean_macro_family_mse",
        "validation_mask_mean_worst_family_mse",
    }:
        raise ValueError(
            "training.selection_metric must be a supported validation-mask aggregate"
        )
    for key in ("deterministic_algorithms", "deterministic_warn_only"):
        if not isinstance(config["training"].get(key), bool):
            raise ValueError(f"training.{key} must be boolean")
    minimum_normal = config["audit"].get("minimum_normal_graphs")
    if not isinstance(minimum_normal, dict) or set(minimum_normal) != expected_splits:
        raise ValueError(
            f"audit.minimum_normal_graphs must contain exactly {sorted(expected_splits)}"
        )
    if any(int(value) < 1 for value in minimum_normal.values()):
        raise ValueError("Every minimum normal graph count must be positive")
    if int(config["audit"].get("minimum_test_attack_graphs", 0)) < 1:
        raise ValueError("minimum_test_attack_graphs must be positive")
    for key in (
        "minimum_target_scalars_per_degree_stratum_across_masks",
        "minimum_target_graphs_per_degree_stratum_across_masks",
    ):
        if int(config["audit"].get(key, 0)) < 0:
            raise ValueError(f"audit.{key} must be nonnegative")
    reciprocal_limit = float(
        config["audit"].get(
            "maximum_reciprocal_timing_exact_match_exposure_rate", -1.0
        )
    )
    if not 0.0 <= reciprocal_limit <= 1.0:
        raise ValueError(
            "audit.maximum_reciprocal_timing_exact_match_exposure_rate must be in [0, 1]"
        )
    for key in (
        "minimum_graph_gain_over_local",
        "minimum_graph_gain_over_set_context",
        "minimum_graph_gain_over_shuffled_graph",
    ):
        value = float(config["audit"].get(key, -1.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"audit.{key} must be in [0, 1]")
    return config


def config_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def preprocessing_code_fingerprint(project_root: str | Path | None = None) -> str:
    """Hash every source file that can change prepared graph artifacts."""

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    package_root = root / "src" / "fedxmag"
    relative_paths = [
        (package_root / name).relative_to(root)
        for name in (
            "config.py",
            "masking.py",
            "preflight.py",
            "schema.py",
            "serialization.py",
        )
    ]
    relative_paths.extend(
        path.relative_to(root) for path in (package_root / "preprocessing").rglob("*.py")
    )
    relative_paths.append(Path("scripts/prepare_xiiotid.py"))
    digest = hashlib.sha256()
    for relative_path in sorted(relative_paths, key=lambda item: item.as_posix()):
        digest.update(relative_path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative_path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def training_code_fingerprint(project_root: str | Path | None = None) -> str:
    """Hash every source file that can change M0 fitting, controls, or evaluation."""

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    package_root = root / "src" / "fedxmag"
    paths = [
        package_root / "config.py",
        package_root / "losses.py",
        package_root / "masking.py",
        package_root / "schema.py",
        package_root / "serialization.py",
        root / "scripts" / "run_pretext_audit.py",
    ]
    paths.extend((package_root / "models").rglob("*.py"))
    paths.extend((package_root / "training").rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
