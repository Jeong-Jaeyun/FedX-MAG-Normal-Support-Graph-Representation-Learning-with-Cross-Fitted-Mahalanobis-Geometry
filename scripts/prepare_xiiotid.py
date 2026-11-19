"""Prepare provenance-rich, leakage-safe X-IIoTID Stage-1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fedxmag.config import (
    config_fingerprint,
    feature_groups_from_config,
    load_stage1_config,
    preprocessing_code_fingerprint,
)
from fedxmag.masking import FamilyMasker
from fedxmag.preflight import build_preflight_report
from fedxmag.preprocessing import (
    FeatureStandardizer,
    build_static_graphs,
    chronological_window_aligned_split,
    read_xiiotid_events,
)
from fedxmag.serialization import save_graph_samples, schema_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_m0.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _ensure_output_is_safe(output_dir: Path, names: list[str], overwrite: bool) -> None:
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing and not overwrite:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing artifacts: {rendered}")


def main() -> None:
    args = parse_args()
    config = load_stage1_config(args.config)
    groups = feature_groups_from_config(config)
    split_artifact_names = [
        "train_normal.pt",
        "select_normal.pt",
        "audit_normal.pt",
        "calibration_normal.pt",
        "test_normal.pt",
        "test_attack.pt",
    ]
    artifact_names = split_artifact_names + [
        "standardizer.pt",
        "manifest.json",
        "preflight.json",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_output_is_safe(args.output_dir, artifact_names, args.overwrite)
    data_config = config["data"]
    window_seconds = int(data_config["window_seconds"])
    if int(data_config["window_stride_seconds"]) != window_seconds:
        raise ValueError("The current static builder requires stride == window size")

    read_result = read_xiiotid_events(
        args.input,
        deduplicate_exact_rows=bool(data_config["deduplicate_exact_rows"]),
    )
    event_splits = chronological_window_aligned_split(
        read_result.events,
        {key: float(value) for key, value in data_config["split_fractions"].items()},
        window_seconds=window_seconds,
        session_split_strategy=str(data_config["session_split_strategy"]),
    )
    timing_profile = str(
        config["features"].get("timing_profile", "legacy_record_iat_v2")
    )
    timestamp_resolution_seconds = int(
        config["features"].get("timestamp_resolution_seconds", 1)
    )
    graph_splits = {
        name: build_static_graphs(
            selected,
            window_seconds=window_seconds,
            groups=groups,
            timing_profile=timing_profile,
            timestamp_resolution_seconds=timestamp_resolution_seconds,
        )
        for name, selected in event_splits.items()
    }
    normal = {
        name: [graph for graph in graphs if graph.label == 0]
        for name, graphs in graph_splits.items()
    }
    required_normal = ("train", "select", "audit", "calibration", "test")
    empty_normal = [name for name in required_normal if not normal[name]]
    test_attacks = [graph for graph in graph_splits["test"] if graph.label == 1]
    if empty_normal or not test_attacks:
        raise ValueError(
            "Required evaluation support is empty: "
            f"normal={empty_normal}, test_attack={len(test_attacks)}. "
            "Inspect temporal/session coverage; do not backfill attacks from earlier splits."
        )

    masking_config = config["masking"]
    masker = FamilyMasker(
        attribute_budget=float(masking_config["attribute_budget"]),
        family_mask_probability=float(masking_config["family_mask_probability"]),
        minimum_visible_per_entity=int(masking_config["minimum_visible_per_entity"]),
        minimum_visible_per_graph=int(masking_config["minimum_visible_per_graph"]),
        groups=groups,
    )
    preflight = build_preflight_report(
        graph_splits, read_result.report, args.input, config, masker, groups
    )
    preprocessing_hash = preprocessing_code_fingerprint()
    preflight["preprocessing_code_sha256"] = preprocessing_hash
    (args.output_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True), encoding="utf-8"
    )
    if preflight["status"] != "pass":
        raise RuntimeError(f"Preflight failed: {preflight['blockers']}")

    scaler = FeatureStandardizer.fit(normal["train"])
    output = {
        "train_normal": scaler.transform_many(normal["train"], groups),
        "select_normal": scaler.transform_many(normal["select"], groups),
        "audit_normal": scaler.transform_many(normal["audit"], groups),
        "calibration_normal": scaler.transform_many(normal["calibration"], groups),
        "test_normal": scaler.transform_many(normal["test"], groups),
        "test_attack": scaler.transform_many(test_attacks, groups),
    }
    earlier_attacks_discarded = sum(
        graph.label == 1
        for name, graphs in graph_splits.items()
        if name != "test"
        for graph in graphs
    )
    manifest = {
        "dataset": "X-IIoTID",
        "window_seconds": window_seconds,
        "split_before_windowing": True,
        "split_is_label_independent": True,
        "deduplicate_exact_rows": data_config["deduplicate_exact_rows"],
        "session_split_strategy": data_config["session_split_strategy"],
        "attack_use": "test_partition_only",
        "earlier_attack_graphs_discarded": earlier_attacks_discarded,
        "counts": {name: len(samples) for name, samples in output.items()},
        "features": {
            "timing_profile": timing_profile,
            "timestamp_resolution_seconds": timestamp_resolution_seconds,
            "node": groups.node_names,
            "edge": groups.edge_names,
        },
        "schema_sha256": schema_fingerprint(groups),
        "config_sha256": config_fingerprint(config),
        "preprocessing_code_sha256": preprocessing_hash,
        "preflight_status": preflight["status"],
        "semantic_note": (
            "Timing is resolution-aware connection-record temporal structure computed "
            "only from the row's observed Src-to-Dst orientation. Spacing uses distinct "
            "timestamp bins; within-bin concurrency is represented by collision and "
            "multiplicity statistics. Synthesized reverse observations are distribution-"
            "only. No feature is raw packet IAT."
            if timing_profile == "distinct_second_multiplicity_v3"
            else (
                "Timing is resolution-aware connection-record temporal structure computed "
                "only from the row's observed Src-to-Dst orientation. Spacing uses distinct "
                "timestamp bins; within-bin concurrency is represented once by collision "
                "fraction. Synthesized reverse observations are distribution-only. No "
                "feature is raw packet IAT."
                if timing_profile == "distinct_second_collision_v3"
                else
                "Timing is connection-record temporal spacing computed only from the row's "
                "observed Src-to-Dst orientation; synthesized reverse observations are "
                "distribution-only. No feature is raw packet IAT."
            )
        ),
    }

    for name, samples in output.items():
        save_graph_samples(
            args.output_dir / f"{name}.pt",
            samples,
            overwrite=args.overwrite,
            groups=groups,
        )
    torch.save(scaler.state_dict(), args.output_dir / "standardizer.pt")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
