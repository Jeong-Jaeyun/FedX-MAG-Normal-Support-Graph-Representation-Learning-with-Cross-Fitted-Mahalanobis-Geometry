"""Build, replay, commission, and use the raw-node + flow-context kNN detector.

The method and its fixed settings (k=20, top-3, select-normal q95, max fusion)
come from the 2026-10-26 root-cause study. Build opens only train, selection,
and calibration normal windows. The two replay phases rescore consumed X-IIoTID
data and must reproduce the archived exploratory scores exactly; they are
software checks, not new evidence. Commissioning a new environment requires
independently vetted normal windows for both reference and calibration.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_entity_detector import write_json

from fedxmag.config import feature_groups_from_config, load_stage1_config
from fedxmag.confirmation import sha256_file
from fedxmag.flow_context import (
    FLOW_FEATURES,
    FLOW_VARIANTS,
    flow_table,
    flows_for,
    read_flow_records,
    window_key,
)
from fedxmag.fusion_detector import (
    CHANNELS,
    DEFAULT_K,
    FPR_TARGETS,
    METHOD,
    RawFlowDetector,
    TrafficWindows,
)
from fedxmag.m1.detection import detection_summary
from fedxmag.preprocessing.scaling import FeatureStandardizer
from fedxmag.schema import FeatureGroups, GraphSample
from fedxmag.serialization import load_graph_samples

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data/processed/xiiotid_stage1_v3b_development"
STUDY = ROOT / "output/root_cause_20261026"
# Format-2 artifacts (flow variant, calibration coverage). output/fusion_detector holds
# the original format-1 build, which still loads with unknown calibration coverage.
OUTPUT = ROOT / "output/fusion_detector_v2"
RAW_CSV = ROOT / "data/X-IIoTID/X-IIoTID dataset.csv"
BASE_CONFIG = ROOT / "configs/stage1_m0_v3_development.yaml"
# Archive order: capture-fold indices refer to the concatenation of these roles.
ROLES = {
    "train": PREPARED / "train_normal.pt",
    "calibration": PREPARED / "calibration_normal.pt",
    "test_normal": PREPARED / "test_normal.pt",
    "test_attack": PREPARED / "test_attack.pt",
    "audit_normal": PREPARED / "audit_normal.pt",
    "audit_attack": ROOT / "data/processed/xiiotid_stage1_v3b_audit_band/audit_attack.pt",
    "selection": PREPARED / "select_normal.pt",
}
SOURCES = (
    "src/fedxmag/flow_context.py",
    "src/fedxmag/fusion_detector.py",
    "src/fedxmag/local_support.py",
    "scripts/run_fusion_detector.py",
)
TOLERANCE = 1e-9


def verified_inputs() -> dict[str, str]:
    """Bind to the exact inputs of the root-cause study and the frozen standardizer."""

    inputs = json.loads((STUDY / "validation_manifest.json").read_text())["inputs"]
    if sha256_file(RAW_CSV) != inputs["raw_sha256"]:
        raise ValueError("The raw X-IIoTID table differs from the root-cause study")
    for role, path in ROLES.items():
        if sha256_file(path) != inputs[role]:
            raise ValueError(f"Prepared windows differ from the root-cause study: {role}")
    registration = json.loads(
        (ROOT / "output/m0/xiiotid_stage1_v3b_confirmation/registration.json").read_text()
    )
    standardizer = sha256_file(PREPARED / "standardizer.pt")
    if standardizer != registration["data_file_sha256"]["standardizer.pt"]:
        raise ValueError("The frozen v3b graph standardizer changed")
    return {**inputs, "graph_standardizer": standardizer}


def load_windows(
    roles: dict[str, Path], groups: FeatureGroups, raw_csv: Path
) -> tuple[dict[str, list[GraphSample]], dict[str, list[torch.Tensor]], dict[str, list]]:
    samples = {role: load_graph_samples(path, groups) for role, path in roles.items()}
    wanted = {window_key(sample) for graphs in samples.values() for sample in graphs}
    print(json.dumps({"phase": "reading_flow_records", "windows": len(wanted)}), flush=True)
    table = flow_table(read_flow_records(raw_csv), wanted)
    flows, keys = {}, {}
    for role, graphs in samples.items():
        flows[role], keys[role] = flows_for(graphs, table)
    return samples, flows, keys


def max_difference(actual: np.ndarray, expected: torch.Tensor) -> float:
    if actual.shape != tuple(expected.shape):
        raise ValueError("Archived score alignment differs")
    return float(np.max(np.abs(actual - expected.numpy()))) if actual.size else 0.0


def build(output: Path, flow_variant: str = "all") -> None:
    destination = output / "detector_source.pt"
    if destination.exists() or (output / "build_report.json").exists():
        raise FileExistsError("Detector output already exists; choose a new --output-dir")
    inputs = verified_inputs()
    groups = feature_groups_from_config(load_stage1_config(BASE_CONFIG))
    fit_roles = {role: ROLES[role] for role in ("train", "selection", "calibration")}
    samples, flows, _ = load_windows(fit_roles, groups, RAW_CSV)
    windows = {role: TrafficWindows(samples[role], flows[role]) for role in fit_roles}
    provenance = {
        "status": "exploratory_candidate_not_independently_confirmed",
        "graph_input_contract": "v3b graphs transformed by the frozen v3b standardizer",
        "inputs": {
            "raw_sha256": inputs["raw_sha256"],
            "graph_standardizer_sha256": inputs["graph_standardizer"],
            **{f"{role}_sha256": inputs[role] for role in fit_roles},
        },
        "source_code": {name: sha256_file(ROOT / name) for name in SOURCES},
    }
    detector = RawFlowDetector.fit(
        groups,
        windows["train"],
        windows["selection"],
        windows["calibration"],
        k=DEFAULT_K,
        flow_variant=flow_variant,
        provenance=provenance,
    )
    output.mkdir(parents=True, exist_ok=True)
    detector.save(destination)
    report = {
        "method": METHOD,
        "settings": {
            "k": detector.k,
            "flow_variant": detector.flow_variant,
            "aggregation": "mean_of_min_3_highest_entity_distances",
            "fusion": "max of channel scores divided by select-normal q95",
            "flow_features": list(FLOW_FEATURES),
            "primary_fpr_target": 0.05,
        },
        "counts": {
            **{f"{role}_windows": len(windows[role]) for role in fit_roles},
            "node_reference_rows": len(detector.node_reference),
            "flow_reference_rows": len(detector.flow_reference),
        },
        "scales": detector.scales,
        "thresholds": detector.thresholds,
        "artifact": destination.name,
        "sha256": sha256_file(destination),
        "provenance": provenance,
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_json(output / "build_report.json", report)
    print(json.dumps({"built": destination.name, "thresholds": detector.thresholds}), flush=True)


def replay(output: Path) -> None:
    destination, scores_path = output / "replay_report.json", output / "replay_scores.pt"
    if destination.exists() or scores_path.exists():
        raise FileExistsError("Replay output already exists")
    verified_inputs()
    build_report = json.loads((output / "build_report.json").read_text())
    artifact = output / build_report["artifact"]
    if sha256_file(artifact) != build_report["sha256"]:
        raise ValueError("Detector artifact changed after build")
    detector = RawFlowDetector.load(artifact)
    samples, flows, _ = load_windows(ROLES, detector.groups, RAW_CSV)
    cache = torch.load(STUDY / "flow_context_cache.pt", weights_only=True)
    if tuple(cache["feature_names"]) != FLOW_FEATURES:
        raise ValueError("Archived flow features differ")
    flow_difference = max(
        float((actual - expected).abs().max())
        for role in ROLES
        for actual, expected in zip(flows[role], cache["features"][role], strict=True)
    )
    archives = {
        "fusion": torch.load(STUDY / "generalization_scores.pt", weights_only=True)[
            "raw_node_plus_flow_knn"
        ],
        "raw_node": torch.load(STUDY / "information_loss_scores.pt", weights_only=True)[
            "v3b_raw_node_knn"
        ],
        "flow": torch.load(STUDY / "context_scores.pt", weights_only=True)["all_flow_knn"],
    }
    scores: dict[str, dict[str, np.ndarray]] = {name: {} for name in ("fusion", *CHANNELS)}
    differences: dict[str, float] = {}
    for role in ROLES:
        if role == "train":
            continue
        fused, channels, _ = detector.score(TrafficWindows(samples[role], flows[role]))
        for name, values in (("fusion", fused), *channels.items()):
            scores[name][role] = values
            if role in archives[name]:
                differences[f"{name}|{role}"] = max_difference(values, archives[name][role])
    np.testing.assert_array_equal(scores["fusion"]["calibration"], detector.calibration_scores)
    report: dict[str, Any] = {
        "status": "replay_of_consumed_data_not_new_evidence",
        "method": METHOD,
        "flow_extraction_max_absolute_difference": flow_difference,
        "archive_max_absolute_score_difference": differences,
        "bands": {},
    }
    for band in ("test", "audit"):
        report["bands"][band] = {
            "counts": {r: len(samples[f"{band}_{r}"]) for r in ("normal", "attack")},
            **{
                name: detection_summary(
                    values["calibration"],
                    values[f"{band}_normal"],
                    values[f"{band}_attack"],
                    FPR_TARGETS,
                )
                for name, values in scores.items()
            },
        }
        fusion = report["bands"][band]["fusion"]
        print(
            json.dumps(
                {"band": band, "auroc": fusion["auroc"], **fusion["at_calibrated_fpr"]["0.05"]}
            ),
            flush=True,
        )
    report["archive_replay_pass"] = bool(
        flow_difference == 0.0
        and len(differences) == 3 * (len(ROLES) - 1) - 1  # raw archive lacks selection
        and max(differences.values()) <= TOLERANCE
    )
    with scores_path.open("xb") as stream:
        torch.save(
            {n: {r: torch.from_numpy(v) for r, v in s.items()} for n, s in scores.items()}, stream
        )
    write_json(destination, report)
    if not report["archive_replay_pass"]:
        raise RuntimeError("The reusable detector did not reproduce the archived scores")


def raw_graphs(samples: list[GraphSample]) -> list[GraphSample]:
    """Invert the frozen standardizer so capture folds can refit graph scaling."""

    old = FeatureStandardizer.from_state_dict(
        torch.load(PREPARED / "standardizer.pt", weights_only=True)
    )
    restored = []
    for graph in samples:
        fields = {}
        for kind in ("node", "edge"):
            values = getattr(graph, f"{kind}_features").double()
            values = (
                values * getattr(old, f"{kind}_scale").double()
                + getattr(old, f"{kind}_mean").double()
            )
            fields[f"{kind}_features"] = values.masked_fill(~getattr(graph, f"{kind}_available"), 0)
        restored.append(replace(graph, **fields))
    return restored


def commission_replay(output: Path) -> None:
    destination = output / "commissioning_replay.json"
    if destination.exists():
        raise FileExistsError(destination)
    verified_inputs()
    groups = feature_groups_from_config(load_stage1_config(BASE_CONFIG))
    samples, flows_by_role, _ = load_windows(ROLES, groups, RAW_CSV)
    raw = raw_graphs([graph for role in ROLES for graph in samples[role]])
    flows = [matrix for role in ROLES for matrix in flows_by_role[role]]
    capture = torch.load(STUDY / "candidate_capture_scores.pt", weights_only=True)
    commissioning = torch.load(STUDY / "normal_commissioning_scores.pt", weights_only=True)
    archived_metrics = json.loads((STUDY / "normal_commissioning.json").read_text())["folds"]
    report: dict[str, Any] = {
        "status": "replay_of_consumed_data_not_new_evidence",
        "condition": "target reference and calibration are ground-truth normal (vetted) windows",
        "folds": {},
    }
    worst = 0.0
    for fold, saved in capture.items():
        indices = {role: values.tolist() for role, values in saved["indices"].items()}
        scaler = FeatureStandardizer.fit([raw[i] for i in indices["train"]])

        def windows(items: list[int], scaler=scaler) -> TrafficWindows:
            graphs = scaler.transform_many([raw[i] for i in items], groups)
            return TrafficWindows(graphs, [flows[i] for i in items])

        source = RawFlowDetector.fit(
            groups,
            windows(indices["train"]),
            windows(indices["selection"]),
            windows(indices["calibration"]),
            provenance={"capture_fold": fold},
        )
        differences = {}
        for role in ("selection", "calibration", "normal", "attack"):
            fused, channels, _ = source.score(windows(indices[role]))
            for name, values in (("fusion", fused), *channels.items()):
                differences[f"source|{name}|{role}"] = max_difference(
                    values, saved["scores"][name][role]
                )
        entry: dict[str, Any] = {"differences": differences}
        if fold in commissioning:
            target = {r: v.tolist() for r, v in commissioning[fold]["target_indices"].items()}
            adapted = source.commission(
                windows(target["reference"]),
                windows(target["calibration"]),
                environment=f"capture_fold_{fold}",
            )
            roles = {
                "source_calibration": indices["calibration"],
                **{r: target[r] for r in ("calibration", "normal", "attack")},
            }
            arms: dict[str, dict[str, np.ndarray]] = {}
            for arm, detector in (("fixed_reference", source), ("reference_added", adapted)):
                arms[arm] = {}
                for role, items in roles.items():
                    fused, channels, _ = detector.score(windows(items))
                    arms[arm][role] = fused
                    archive = commissioning[fold]["scores"][arm]
                    for name, values in (("fusion", fused), *channels.items()):
                        differences[f"{arm}|{name}|{role}"] = max_difference(
                            values, archive[name][role]
                        )
            np.testing.assert_array_equal(adapted.calibration_scores, arms["reference_added"][
                "calibration"])
            entry["metrics"] = {}
            for arm, values in arms.items():
                for calibration in ("source_calibration", "calibration"):
                    key = f"{arm}__{calibration}"
                    summary = detection_summary(
                        values[calibration], values["normal"], values["attack"], [0.05]
                    )
                    expected = archived_metrics[fold]["metrics"][key]["fusion"]
                    entry["metrics"][key] = {
                        "auroc": summary["auroc"],
                        **{m: summary["at_calibrated_fpr"]["0.05"][m] for m in ("tpr", "fpr")},
                        "matches_archived_report": summary == expected,
                    }
            entry["commissioned_thresholds"] = adapted.thresholds
            entry["counts"] = {role: len(items) for role, items in roles.items()}
        entry["max_absolute_score_difference"] = max(differences.values())
        worst = max(worst, entry["max_absolute_score_difference"])
        report["folds"][fold] = entry
        print(
            json.dumps(
                {
                    "capture_fold": fold,
                    "max_abs_diff": entry["max_absolute_score_difference"],
                    "commissioned": entry.get("metrics", {}).get(
                        "reference_added__calibration"
                    ),
                }
            ),
            flush=True,
        )
    report["max_absolute_score_difference"] = worst
    report["archive_replay_pass"] = bool(
        worst <= TOLERANCE
        and all(
            item["matches_archived_report"]
            for entry in report["folds"].values()
            for item in entry.get("metrics", {}).values()
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(destination, report)
    if not report["archive_replay_pass"]:
        raise RuntimeError("Commissioning did not reproduce the archived study")


def score(args: argparse.Namespace) -> None:
    if args.predictions.exists():
        raise FileExistsError(f"Predictions already exist: {args.predictions}")
    detector = RawFlowDetector.load(args.detector)
    samples, flows, keys = load_windows({"input": args.input}, detector.groups, args.raw_csv)
    windows = TrafficWindows(samples["input"], flows["input"])
    result = {
        "method": METHOD,
        "environment": detector.environment,
        "detector_sha256": sha256_file(args.detector),
        "input_sha256": sha256_file(args.input),
        "raw_csv_sha256": sha256_file(args.raw_csv),
        "primary_fpr_target": 0.05,
        "thresholds": detector.thresholds,
        "windows": detector.predict(windows, keys["input"]),
    }
    write_json(args.predictions, result)
    print(
        json.dumps(
            {
                "windows": len(windows),
                **{
                    f"windows_outside_{role}": sum(
                        item[f"capture_in_{role}"] is False for item in result["windows"]
                    )
                    for role in ("reference", "calibration")
                },
            }
        )
    )


def commission(args: argparse.Namespace) -> None:
    if args.artifact.exists():
        raise FileExistsError(f"Detector already exists: {args.artifact}")
    detector = RawFlowDetector.load(args.detector)
    roles = {"reference": args.reference}
    if args.calibration is not None:
        roles["calibration"] = args.calibration
    samples, flows, _ = load_windows(roles, detector.groups, args.raw_csv)
    windows = {role: TrafficWindows(samples[role], flows[role]) for role in roles}
    adapted = detector.commission(
        windows["reference"],
        windows.get("calibration"),
        environment=args.environment,
        blocks=args.blocks,
        record={
            "source_detector_sha256": sha256_file(args.detector),
            **{f"{role}_sha256": sha256_file(path) for role, path in roles.items()},
            "raw_csv_sha256": sha256_file(args.raw_csv),
        },
    )
    adapted.save(args.artifact)
    print(json.dumps({"environment": adapted.environment, "thresholds": adapted.thresholds}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "replay", "commission-replay", "score",
                                          "commission"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--detector", type=Path, default=OUTPUT / "detector_source.pt")
    parser.add_argument("--raw-csv", type=Path, default=RAW_CSV)
    parser.add_argument(
        "--flow-variant", choices=tuple(FLOW_VARIANTS), default="all", help="build: flow channel"
    )
    parser.add_argument("--input", type=Path, help="score: graphs to score")
    parser.add_argument("--predictions", type=Path, help="score: new JSON output file")
    parser.add_argument("--reference", type=Path, help="commission: vetted normal reference")
    parser.add_argument("--calibration", type=Path, help="commission: disjoint vetted normal")
    parser.add_argument(
        "--blocks", type=int, help="commission: time-block cross-fitted calibration instead"
    )
    parser.add_argument("--environment", help="commission: environment name")
    parser.add_argument("--artifact", type=Path, help="commission: new detector file")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase == "build":
        build(args.output_dir, args.flow_variant)
    elif args.phase == "replay":
        replay(args.output_dir)
    elif args.phase == "commission-replay":
        commission_replay(args.output_dir)
    elif args.phase == "score":
        if args.input is None or args.predictions is None:
            parser.error("score requires --input and --predictions")
        score(args)
    else:
        if None in (args.reference, args.environment, args.artifact):
            parser.error("commission requires --reference, --environment, and --artifact")
        if (args.calibration is None) == (args.blocks is None):
            parser.error("commission requires exactly one of --calibration or --blocks")
        commission(args)


if __name__ == "__main__":
    main()
