"""Pre-registered TON_IoT validation of the raw-node + flow detector and commissioning.

``register`` binds the protocol, code, census, and network tables before any TON_IoT
window is featurized. ``featurize`` builds source windows and target attack and
background samples once and runs the protocol's integrity checks. ``evaluate``
scores everything once, refuses to run if a bound file changed, and applies the
fixed claims of configs/ton_iot_validation_protocol.yaml.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from run_cross_calibration_check import binomial_upper_tail
from run_entity_detector import write_json
from run_fusion_detector import BASE_CONFIG

from fedxmag.config import feature_groups_from_config, load_stage1_config
from fedxmag.confirmation import sha256_file
from fedxmag.flow_context import FLOW_VARIANTS, WINDOW_NS
from fedxmag.fusion_detector import (
    CHANNELS,
    DEFAULT_K,
    RawFlowDetector,
    TrafficWindows,
    calibrated_thresholds,
)
from fedxmag.m1.detection import auroc
from fedxmag.preprocessing.graph_builder import _split_ordered_items
from fedxmag.preprocessing.scaling import FeatureStandardizer
from fedxmag.schema import GraphSample
from fedxmag.serialization import load_graph_samples, save_graph_samples
from fedxmag.ton_iot import capture_of, read_ton_records, session_key

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/ton_iot_validation_protocol.yaml"
DATA = ROOT / "data/TON-IoT/Processed_datasets/Processed_Network_dataset"
CENSUS = ROOT / "output/ton_iot_census"
OUTPUT = ROOT / "output/ton_iot_validation_20261027"
CODE = (
    "src/fedxmag/ton_iot.py",
    "src/fedxmag/flow_context.py",
    "src/fedxmag/fusion_detector.py",
    "src/fedxmag/local_support.py",
    "src/fedxmag/m1/detection.py",
    "src/fedxmag/m1/entity.py",
    "src/fedxmag/preprocessing/events.py",
    "src/fedxmag/preprocessing/graph_builder.py",
    "src/fedxmag/preprocessing/scaling.py",
    "src/fedxmag/schema.py",
    "configs/stage1_m0_v3_development.yaml",
    "scripts/census_ton_iot.py",
    "scripts/run_ton_iot_validation.py",
)
SOURCE_ROLES = ("train", "selection", "audit", "calibration", "test")
SOURCE_FRACTIONS = (0.50, 0.10, 0.10, 0.15, 0.15)
VIEWS = ("source", "attack", "background")
PRIMARY = "0.05"
ENVIRONMENT = "ton_iot_attack_period"
VOLUME_KEYS = {"record_count": "records", "endpoint_count": "endpoint_keys"}


def tables(data: Path) -> list[Path]:
    return sorted(data.glob("Network_dataset_*.csv"), key=lambda p: int(p.stem.split("_")[-1]))


def bindings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": {str(args.protocol): sha256_file(args.protocol)},
        "code": {name: sha256_file(ROOT / name) for name in CODE},
        "census": {
            name: sha256_file(args.census / name) for name in ("layout.json", "support.json")
        },
        "tables": {path.name: sha256_file(path) for path in tables(args.data)},
    }


def check_registration(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    registration = json.loads((args.output / "registration.json").read_text())
    current = bindings(args)
    changed = [key for key, value in current.items() if registration[key] != value]
    if changed:
        raise ValueError(f"Registered files changed: {changed}")
    return yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8")), registration


def register(args: argparse.Namespace) -> None:
    path = args.output / "registration.json"
    if path.exists():
        raise FileExistsError("This validation is already registered")
    layout = json.loads((args.census / "layout.json").read_text())
    current = bindings(args)
    if {item["file"]: item["sha256"] for item in layout["files"]} != current["tables"]:
        raise ValueError("The network tables differ from the layout census")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "status": "registered_before_featurization_and_any_score",
            "registered_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            **current,
        },
    )
    print(json.dumps({"registered": str(path)}))


def _settings() -> tuple[Any, dict[str, Any]]:
    config = load_stage1_config(BASE_CONFIG)
    features = config["features"]
    return feature_groups_from_config(config), {
        "timing_profile": features["timing_profile"],
        "timestamp_resolution_seconds": int(features["timestamp_resolution_seconds"]),
    }


def featurize(args: argparse.Namespace) -> None:
    views_dir = args.output / "views"
    report_path = args.output / "featurize_report.json"
    if report_path.exists() or views_dir.exists():
        raise FileExistsError("Views already exist")
    protocol, _ = check_registration(args)
    roles = protocol["roles"]
    started = time.perf_counter()
    groups, settings = _settings()
    records = read_ton_records(tables(args.data))
    support = json.loads((args.census / "support.json").read_text())
    census_rows = {"rows_total": "total", "exact_duplicate": "exact_duplicate",
                   "bad_timestamp": "bad_timestamp", "missing_endpoint": "missing_endpoint",
                   "self_relation": "self_relation", "kept": "kept"}
    checks: dict[str, bool] = {
        "row_counts_equal_census": all(
            records.report[mine] == support["rows"].get(theirs, 0)
            for mine, theirs in census_rows.items()
        )
    }
    by_capture: dict[str, list[int]] = defaultdict(list)
    windows = records.windows()
    for minute in sorted(windows):
        by_capture[capture_of(minute * WINDOW_NS)].append(minute)
    observed = {
        capture: {
            "windows": len(minutes),
            "normal_windows": sum(not records.attack[windows[m]].any() for m in minutes),
            "attack_windows": sum(bool(records.attack[windows[m]].any()) for m in minutes),
        }
        for capture, minutes in by_capture.items()
    }
    checks["window_counts_equal_census"] = observed == {
        capture: {key: item[key] for key in ("windows", "normal_windows", "attack_windows")}
        for capture, item in support["captures"].items()
    }
    if not all(checks.values()):
        raise RuntimeError(f"Integrity check failed before featurization: {checks}")

    def build(rows: np.ndarray, view: str, extra: dict[str, Any]) -> tuple[GraphSample, Any]:
        graph, flows, _ = records.view(rows, groups, **settings)
        attack_rows = rows[records.attack[rows]]
        graph.metadata.update(
            view=view,
            capture=capture_of(int(records.timestamps[rows[0]])),
            records=len(rows),
            attack_records=len(attack_rows),
            attack_endpoints=sorted(
                {records.endpoints[i] for i in records.sources[attack_rows]}
                | {records.endpoints[i] for i in records.destinations[attack_rows]}
            ),
            **extra,
        )
        return graph, flows

    built: dict[str, tuple[list[GraphSample], list[torch.Tensor]]] = {v: ([], []) for v in VIEWS}
    for capture in roles["source_captures"]:
        minutes = by_capture[capture]
        for role, members in zip(
            SOURCE_ROLES, _split_ordered_items(minutes, SOURCE_FRACTIONS), strict=True
        ):
            for minute in members:
                graph, flows = build(windows[minute], "source", {"role": role})
                built["source"][0].append(graph)
                built["source"][1].append(flows)
    for capture in roles["target_captures"]:
        for minute in by_capture[capture]:
            rows = windows[minute]
            attack = records.attack[rows]
            natural = not attack.any()
            if not attack.all():
                graph, flows = build(rows[~attack], "background", {"natural_normal": natural})
                built["background"][0].append(graph)
                built["background"][1].append(flows)
            if not natural:
                graph, flows = build(rows, "attack", {"natural_normal": False})
                built["attack"][0].append(graph)
                built["attack"][1].append(flows)
        print(json.dumps({"featurized": capture, "seconds": round(time.perf_counter() - started)}),
              flush=True)

    source_graphs = built["source"][0]
    checks["source_windows_all_normal"] = all(g.label == 0 for g in source_graphs)
    checks["background_samples_label_0_only"] = all(
        g.label == 0 and g.metadata["attack_records"] == 0 for g in built["background"][0]
    )
    checks["attack_samples_have_attack_records"] = all(
        g.label == 1 and g.metadata["attack_records"] > 0 for g in built["attack"][0]
    )
    checks["flows_finite"] = all(
        bool(torch.isfinite(f).all()) for view in VIEWS for f in built[view][1]
    )
    if not all(checks.values()):
        raise RuntimeError(f"Integrity check failed after featurization: {checks}")
    views_dir.mkdir(parents=True)
    hashes = {}
    for view in VIEWS:
        graphs, flows = built[view]
        save_graph_samples(views_dir / f"{view}_graphs.pt", graphs, groups=groups)
        with (views_dir / f"{view}_flows.pt").open("xb") as stream:
            torch.save(flows, stream)
        for name in (f"{view}_graphs.pt", f"{view}_flows.pt"):
            hashes[name] = sha256_file(views_dir / name)
    write_json(
        report_path,
        {
            "status": "featurized_after_registration_no_scores",
            "ingestion": records.report,
            "integrity_checks": checks,
            "counts": {
                view: {
                    capture: sum(g.metadata["capture"] == capture for g in built[view][0])
                    for capture in sorted({g.metadata["capture"] for g in built[view][0]})
                }
                for view in VIEWS
            },
            "source_roles": {
                role: sum(g.metadata["role"] == role for g in source_graphs)
                for role in SOURCE_ROLES
            },
            "view_sha256": hashes,
            "seconds": round(time.perf_counter() - started),
            "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    print(json.dumps({"featurized": {view: len(built[view][0]) for view in VIEWS}}), flush=True)


def load_views(args: argparse.Namespace, groups) -> dict[str, tuple[list[GraphSample], list]]:
    views_dir = args.output / "views"
    report = json.loads((args.output / "featurize_report.json").read_text())
    for name, digest in report["view_sha256"].items():
        if sha256_file(views_dir / name) != digest:
            raise ValueError(f"Featurized view changed: {name}")
    return {
        view: (
            load_graph_samples(views_dir / f"{view}_graphs.pt", groups),
            torch.load(views_dir / f"{view}_flows.pt", weights_only=True),
        )
        for view in VIEWS
    }


def commissioning_split(
    background: list[GraphSample], attack: list[GraphSample], captures: list[str]
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Per capture: first n//5 background samples reference, next n//5 calibration."""

    split: dict[str, list[int]] = {r: [] for r in ("reference", "calibration", "negative",
                                                   "positive")}
    coverage: dict[str, Any] = {}
    for capture in captures:
        own = sorted(
            (i for i, g in enumerate(background) if g.metadata["capture"] == capture),
            key=lambda i: background[i].metadata["window_start_ns"],
        )
        if len(own) < 10:
            coverage[capture] = {"status": "unsupported_normal_coverage", "background": len(own)}
            continue
        count = len(own) // 5
        reference, calibration = own[:count], own[count : 2 * count]
        cutoff = background[calibration[-1]].metadata["window_start_ns"]
        negative = [
            i for i in own[2 * count :] if background[i].metadata["window_start_ns"] > cutoff
        ]
        positive = [
            i for i, g in enumerate(attack)
            if g.metadata["capture"] == capture and g.metadata["window_start_ns"] > cutoff
        ]
        assert len(negative) == len(own) - 2 * count
        assert max(background[i].metadata["window_start_ns"] for i in reference) < min(
            background[i].metadata["window_start_ns"] for i in calibration
        )
        for role, members in zip(split, (reference, calibration, negative, positive),
                                 strict=True):
            split[role].extend(members)
        coverage[capture] = {
            "status": "supported",
            "reference": len(reference),
            "calibration": len(calibration),
            "negative": len(negative),
            "positive": len(positive),
            "earlier_attack_excluded": sum(
                g.metadata["capture"] == capture and g.metadata["window_start_ns"] <= cutoff
                for g in attack
            ),
        }
    return split, coverage


def at_targets(
    thresholds: dict[str, float | None],
    negative: np.ndarray,
    positive: np.ndarray,
    negative_captures: np.ndarray,
    positive_captures: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "auroc": auroc(negative, positive) if len(negative) and len(positive) else None,
        "negatives": len(negative),
        "positives": len(positive),
        "thresholds": thresholds,
        "at_target": {},
    }
    if len(negative) and len(positive):
        oracle = np.sort(negative)[math.ceil(0.95 * len(negative)) - 1]
        result["oracle_tpr_at_eval_fpr_5pct"] = float(np.mean(positive > oracle))
    for target, threshold in thresholds.items():
        if threshold is None:
            result["at_target"][target] = None
            continue
        tp, fp = int(np.sum(positive > threshold)), int(np.sum(negative > threshold))
        result["at_target"][target] = {
            "tp": tp,
            "fp": fp,
            "tpr": tp / len(positive) if len(positive) else None,
            "fpr": fp / len(negative) if len(negative) else None,
            "binomial_p_fpr_leq_target": binomial_upper_tail(fp, len(negative), float(target)),
        }
    primary = thresholds.get(PRIMARY)
    if primary is not None:
        result["per_capture_at_5pct"] = {
            capture: {
                "fp": int(np.sum(negative[negative_captures == capture] > primary)),
                "negatives": int(np.sum(negative_captures == capture)),
                "tp": int(np.sum(positive[positive_captures == capture] > primary)),
                "positives": int(np.sum(positive_captures == capture)),
                "auroc": (
                    auroc(negative[negative_captures == capture],
                          positive[positive_captures == capture])
                    if np.any(negative_captures == capture)
                    and np.any(positive_captures == capture)
                    else None
                ),
            }
            for capture in sorted(set(negative_captures) | set(positive_captures))
        }
    return result


def safety(summary: dict[str, Any], minimum: int) -> dict[str, Any]:
    pooled = summary["at_target"][PRIMARY]
    eligible = {
        capture: counts["fp"] / counts["negatives"]
        for capture, counts in summary["per_capture_at_5pct"].items()
        if counts["negatives"] >= minimum
    }
    holds = pooled["binomial_p_fpr_leq_target"] >= 0.05 and all(
        value <= 0.10 for value in eligible.values()
    )
    return {"holds": bool(holds), "binomial_p": pooled["binomial_p_fpr_leq_target"],
            "eligible_capture_fpr": eligible}


def tpr_difference(
    arm: dict[str, Any], baseline: dict[str, Any], draws: np.ndarray, captures: list[str],
    levels: tuple[float, float],
) -> dict[str, Any]:
    """Pooled TPR difference at 5% with a capture bootstrap (thresholds held fixed)."""

    def counts(summary: dict[str, Any], key: str) -> np.ndarray:
        per = summary["per_capture_at_5pct"]
        return np.asarray([per.get(c, {}).get(key, 0) for c in captures], dtype=np.float64)

    positives = counts(arm, "positives")
    gained = counts(arm, "tp") - counts(baseline, "tp")
    samples = [float(d @ gained / (d @ positives)) for d in draws if d @ positives]
    return {
        "point": float(gained.sum() / positives.sum()),
        "lower": float(np.quantile(samples, levels[0])),
        "upper": float(np.quantile(samples, levels[1])),
        "interval_levels": list(levels),
        "resamples_used": len(samples),
    }


def volume(graphs: list[GraphSample], members, name: str) -> np.ndarray:
    """log1p of a sample's kept records or endpoints."""

    values = [graphs[i].metadata[VOLUME_KEYS[name]] for i in members]
    sizes = [len(value) if isinstance(value, list) else value for value in values]
    return np.log1p(np.asarray(sizes, dtype=np.float64))


def verdict(safe: bool, difference: dict[str, Any]) -> str:
    if safe and difference["point"] > 0 and difference["lower"] > 0:
        return "supported"
    if safe and difference["point"] > 0:
        return "inconclusive"
    return "not_supported"


def evaluate(args: argparse.Namespace) -> None:
    destination = args.output / "report.json"
    if destination.exists():
        raise FileExistsError(destination)
    protocol, registration = check_registration(args)
    started = time.perf_counter()
    groups, _ = _settings()
    views = load_views(args, groups)
    source_graphs, source_flows = views["source"]
    raw_role = np.asarray([g.metadata["role"] for g in source_graphs])
    scaler = FeatureStandardizer.fit(
        [g for g, role in zip(source_graphs, raw_role, strict=True) if role == "train"]
    )

    def windows(view: str, members: list[int] | np.ndarray) -> TrafficWindows:
        graphs, flows = views[view]
        chosen = scaler.transform_many([graphs[i] for i in members], groups)
        return TrafficWindows(chosen, [flows[i] for i in members])

    source = {role: windows("source", np.flatnonzero(raw_role == role).tolist())
              for role in SOURCE_ROLES}
    background, attack = views["background"][0], views["attack"][0]
    captures = list(protocol["roles"]["target_captures"])
    split, coverage = commissioning_split(background, attack, captures)
    target = {
        "reference": windows("background", split["reference"]),
        "calibration": windows("background", split["calibration"]),
        "negative": windows("background", split["negative"]),
        "positive": windows("attack", split["positive"]),
    }
    negative_caps = np.asarray([background[i].metadata["capture"] for i in split["negative"]])
    positive_caps = np.asarray([attack[i].metadata["capture"] for i in split["positive"]])
    minimum = int(protocol["metrics"]["safety_capture_minimum"])
    boot = protocol["metrics"]["bootstrap"]
    rng = np.random.default_rng(boot["seed"])
    supported = [c for c in captures if coverage[c]["status"] == "supported"]
    draws = rng.multinomial(len(supported), np.full(len(supported), 1 / len(supported)),
                            boot["resamples"])

    report: dict[str, Any] = {
        "status": protocol["protocol"]["status"],
        "protocol_sha256": next(iter(registration["protocol"].values())),
        "coverage": coverage,
        "variants": {},
    }
    saved: dict[str, Any] = {"split": {r: torch.tensor(v) for r, v in split.items()}}
    detectors: dict[str, dict[str, RawFlowDetector]] = {}
    for variant in FLOW_VARIANTS:
        fitted = RawFlowDetector.fit(
            groups, source["train"], source["selection"], source["calibration"],
            k=DEFAULT_K, flow_variant=variant,
        )
        arms = {
            "source_only": fitted,
            "recalibrated": fitted.recalibrate(target["calibration"], environment=ENVIRONMENT),
            "commissioned": fitted.commission(
                target["reference"], target["calibration"], environment=ENVIRONMENT
            ),
        }
        detectors[variant] = arms
        scored = {
            "source": {r: fitted.score(target[r]) for r in ("negative", "positive")},
            "commissioned": {
                r: arms["commissioned"].score(target[r]) for r in ("negative", "positive")
            },
        }
        entry: dict[str, Any] = {"arms": {}}
        saved[variant] = {}
        for name, detector in arms.items():
            scores = scored["commissioned" if name == "commissioned" else "source"]
            negative, positive = scores["negative"][0], scores["positive"][0]
            summary = at_targets(detector.thresholds, negative, positive, negative_caps,
                                 positive_caps)
            summary["calibration_windows"] = len(detector.calibration_scores)
            summary["coverage_flags"] = {
                capture: {
                    "capture_in_reference": session_key(capture) in detector.reference_sessions,
                    "capture_in_calibration": (
                        session_key(capture) in detector.calibration_sessions
                    ),
                }
                for capture in supported
            }
            entry["arms"][name] = summary
            saved[variant][name] = {
                "calibration": torch.from_numpy(detector.calibration_scores.copy()),
                "negative": torch.from_numpy(negative),
                "positive": torch.from_numpy(positive),
            }
        entry["P1_shift"] = {
            "binomial_p": entry["arms"]["source_only"]["at_target"][PRIMARY][
                "binomial_p_fpr_leq_target"
            ],
        }
        entry["P1_shift"]["reproduced"] = bool(entry["P1_shift"]["binomial_p"] < 0.05)
        entry["safety"] = safety(entry["arms"]["commissioned"], minimum)
        entry["commissioned_vs_recalibrated"] = tpr_difference(
            entry["arms"]["commissioned"], entry["arms"]["recalibrated"], draws, supported,
            (0.025, 0.975),
        )
        if variant == "all":
            # S4: each channel with its own calibration thresholds.
            entry["channels"] = {}
            for name in ("recalibrated", "commissioned"):
                calibration = arms[name].score(target["calibration"])[1]
                scores = scored["commissioned" if name == "commissioned" else "source"]
                entry["channels"][name] = {
                    channel: at_targets(
                        calibrated_thresholds(calibration[channel]),
                        scores["negative"][1][channel], scores["positive"][1][channel],
                        negative_caps, positive_caps,
                    )
                    for channel in CHANNELS
                }
            control_scored = scored
        report["variants"][variant] = entry
        print(json.dumps({
            "variant": variant,
            **{n: {k: a["at_target"][PRIMARY] and round(a["at_target"][PRIMARY][k], 4)
                   for k in ("tpr", "fpr")} for n, a in entry["arms"].items()},
            "seconds": round(time.perf_counter() - started),
        }), flush=True)

    control = report["variants"]["all"]
    # Volume baselines share the evaluation samples and calibration roles.
    counts = {}
    for name in VOLUME_KEYS:
        calibration_source = volume(source_graphs, np.flatnonzero(raw_role == "calibration"), name)
        calibration_target = volume(background, split["calibration"], name)
        negative = volume(background, split["negative"], name)
        positive = volume(attack, split["positive"], name)
        counts[name] = {
            "source": at_targets(calibrated_thresholds(calibration_source), negative, positive,
                                 negative_caps, positive_caps),
            "target": at_targets(calibrated_thresholds(calibration_target), negative, positive,
                                 negative_caps, positive_caps),
        }
        saved[name] = {"negative": torch.from_numpy(negative),
                       "positive": torch.from_numpy(positive)}
    report["volume_baselines"] = counts

    # Primary claims (control flow variant).
    commissioning = verdict(control["safety"]["holds"], control["commissioned_vs_recalibrated"])
    beyond_volume = tpr_difference(
        control["arms"]["commissioned"], counts["record_count"]["target"], draws, supported,
        (0.025, 0.975),
    )
    beyond = verdict(control["safety"]["holds"], beyond_volume)
    report["primary"] = {
        "P1_shift_reproduced": control["P1_shift"]["reproduced"],
        "P1_binomial_p": control["P1_shift"]["binomial_p"],
        "P2_safety": control["safety"],
        "P3_commissioned_minus_recalibrated": control["commissioned_vs_recalibrated"],
        "P2_P3_commissioning": commissioning,
        "P4_commissioned_minus_record_count": beyond_volume,
        "P4_beyond_volume": beyond,
        "validated": commissioning == "supported" and beyond == "supported",
    }

    # S1: flow variants against the control, Bonferroni over three comparisons.
    levels = (0.05 / 3 / 2, 1 - 0.05 / 3 / 2)
    candidates = {}
    for variant in protocol["secondary"]["S1_flow_variants"]["candidates"]:
        difference = tpr_difference(
            report["variants"][variant]["arms"]["commissioned"], control["arms"]["commissioned"],
            draws, supported, levels,
        )
        safe = report["variants"][variant]["safety"]["holds"]
        candidates[variant] = {
            "difference_vs_control": difference,
            "safety_holds": safe,
            "qualifies": bool(safe and difference["lower"] > 0),
        }
    qualifying = [v for v, item in candidates.items() if item["qualifies"]]
    report["S1_flow_variants"] = {
        "candidates": candidates,
        "recommended_default": max(
            qualifying, key=lambda v: candidates[v]["difference_vs_control"]["point"]
        ) if qualifying else "all",
    }

    # S2: FPR at 5% by coverage flags on the same evaluation negatives.
    report["S2_coverage_flags"] = {
        name: {
            "capture_in_reference": name == "commissioned",
            "capture_in_calibration": name != "source_only",
            "fpr": control["arms"][name]["at_target"][PRIMARY]["fpr"],
        }
        for name in ("source_only", "recalibrated", "commissioned")
    }
    control_source = detectors["all"]["source_only"]
    report["S2_coverage_flags"]["source_environment"] = {
        role: at_targets(control_source.thresholds, control_source.score(source[role])[0],
                         np.zeros(0), np.asarray(["source"] * len(source[role])),
                         np.asarray([], dtype=str))
        for role in ("audit", "test")
    }

    # S3: per-capture commissioning.
    per_capture = {}
    for capture in supported:
        own = {
            role: [i for i in split[role]
                   if (background if role != "positive" else attack)[i].metadata["capture"]
                   == capture]
            for role in split
        }
        item = coverage[capture] | {"calibration_size_at_least_100": len(own["calibration"]) >= 100}
        if len(own["calibration"]) < 19:
            item["status"] = "unsupported_5pct_target"
            per_capture[capture] = item
            continue
        detector = control_source.commission(
            windows("background", own["reference"]), windows("background", own["calibration"]),
            environment=capture,
        )
        negative = detector.score(windows("background", own["negative"]))[0]
        positive = (detector.score(windows("attack", own["positive"]))[0]
                    if own["positive"] else np.zeros(0))
        item["metrics"] = at_targets(detector.thresholds, negative, positive,
                                     np.asarray([capture] * len(negative)),
                                     np.asarray([capture] * len(positive)))
        per_capture[capture] = item
    report["S3_per_capture_commissioning"] = per_capture

    # S5: attack types; S7: entities; S8: natural normals (control, commissioned).
    commissioned = control["arms"]["commissioned"]
    threshold = commissioned["thresholds"][PRIMARY]
    count_threshold = counts["record_count"]["target"]["thresholds"][PRIMARY]
    positive_scores = control_scored["commissioned"]["positive"][0]
    positive_counts = saved["record_count"]["positive"].numpy()
    types = sorted({t for i in split["positive"] for t in attack[i].metadata["attack_categories"]})
    report["S5_attack_types"] = {}
    for kind in types:
        mask = np.asarray([kind in attack[i].metadata["attack_categories"]
                           for i in split["positive"]])
        report["S5_attack_types"][kind] = {
            "positives": int(mask.sum()),
            "captures": sorted(set(positive_caps[mask])),
            "commissioned_tpr": float(np.mean(positive_scores[mask] > threshold)),
            "record_count_tpr": float(np.mean(positive_counts[mask] > count_threshold)),
        }
    node_scores = control_scored["commissioned"]["positive"][2]["raw_node"]
    labels, values, per_window = [], [], []
    for position, index in enumerate(split["positive"]):
        metadata = attack[index].metadata
        involved = set(metadata["attack_endpoints"])
        own_labels = [key in involved for key in metadata["endpoint_keys"]]
        own_values = node_scores[position].numpy()
        labels.extend(own_labels)
        values.extend(own_values.tolist())
        if any(own_labels) and not all(own_labels):
            flags = np.asarray(own_labels)
            per_window.append(auroc(own_values[~flags], own_values[flags]))
    labels_array, values_array = np.asarray(labels), np.asarray(values)
    report["S7_entities"] = {
        "attack_endpoints": int(labels_array.sum()),
        "other_endpoints": int((~labels_array).sum()),
        "pooled_auroc": auroc(values_array[~labels_array], values_array[labels_array]),
        "mean_window_auroc": float(np.mean(per_window)) if per_window else None,
        "windows_with_both_classes": len(per_window),
    }
    natural = np.asarray([background[i].metadata["natural_normal"] for i in split["negative"]])
    report["S8_natural_normals"] = {
        name: {
            "negatives": int(natural.sum()),
            "fp": int(np.sum(saved["all"][name]["negative"].numpy()[natural]
                             > control["arms"][name]["thresholds"][PRIMARY])),
        }
        for name in ("source_only", "recalibrated", "commissioned")
    }

    # S6: the confounded natural-window comparison (source-only, every attack sample).
    all_attack = windows("attack", list(range(len(attack))))
    source_test = control_source.score(source["test"])[0]
    every = control_source.score(all_attack)[0]
    test_idx = np.flatnonzero(raw_role == "test")
    report["S6_confounded_natural_windows"] = {
        "source_test_normals": len(source_test),
        "attack_windows": len(every),
        "fused_auroc": auroc(source_test, every),
        **{
            f"{name}_auroc": auroc(
                volume(source_graphs, test_idx, name), volume(attack, range(len(attack)), name)
            )
            for name in VOLUME_KEYS
        },
    }
    report["seconds"] = round(time.perf_counter() - started)
    report["finished_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    json.dumps(report, allow_nan=False)  # fail before writing anything
    with (args.output / "scores.pt").open("xb") as stream:
        torch.save(saved, stream)
    write_json(destination, report)
    print(json.dumps({"primary": report["primary"],
                      "S1_default": report["S1_flow_variants"]["recommended_default"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "featurize", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--census", type=Path, default=CENSUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    {"register": register, "featurize": featurize, "evaluate": evaluate}[args.phase](args)


if __name__ == "__main__":
    main()
