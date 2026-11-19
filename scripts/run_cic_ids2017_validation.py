"""Pre-registered CIC-IDS2017 confirmation of fresh calibration and of detection beyond volume.

``register`` binds the protocol, code, census, converted and labelled tables before any
CIC-IDS2017 window is featurized. ``featurize`` builds the Monday source windows and
every Thursday and Friday window (plus label-0 background samples of attack windows)
once and runs the integrity checks. ``evaluate`` scores everything once, refuses to
run if a bound file changed, and applies the fixed claims of
configs/cic_ids2017_validation_protocol.yaml.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from run_cross_calibration_check import binomial_upper_tail
from run_entity_detector import write_json
from run_ton_iot_validation import SOURCE_FRACTIONS, SOURCE_ROLES, _settings

from fedxmag.cic_ids2017 import CAPTURE_OFFSET_SECONDS, DAYS, capture_of, session_key
from fedxmag.confirmation import sha256_file
from fedxmag.flow_context import WINDOW_NS
from fedxmag.fusion_detector import DEFAULT_K, RawFlowDetector, TrafficWindows
from fedxmag.m1.detection import auroc, conformal_threshold
from fedxmag.preprocessing.graph_builder import _split_ordered_items
from fedxmag.preprocessing.scaling import FeatureStandardizer
from fedxmag.schema import GraphSample
from fedxmag.serialization import load_graph_samples, save_graph_samples
from fedxmag.ton_iot import read_ton_records

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/cic_ids2017_validation_protocol.yaml"
CONN = ROOT / "data/processed/cic_ids2017_conn"
CENSUS = ROOT / "output/cic_ids2017_census"
OUTPUT = ROOT / "output/cic_ids2017_validation_20261101"
CODE = (
    "src/fedxmag/pcap_conn.py",
    "src/fedxmag/cic_ids2017.py",
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
    "scripts/convert_cic_ids2017.py",
    "scripts/census_cic_ids2017.py",
    "scripts/run_cic_ids2017_validation.py",
    "scripts/run_ton_iot_validation.py",
    "scripts/run_cross_calibration_check.py",
)
CENSUS_FILES = ("layout.json", "label.json", "support.json")
VIEWS = ("source", "window", "background")
TARGET = 0.05
QUANTILE = 1.0 - TARGET
SCORES = ("detector", "record_count", "endpoint_count", "combined")


def bindings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": {str(args.protocol): sha256_file(args.protocol)},
        "code": {name: sha256_file(ROOT / name) for name in CODE},
        "census": {name: sha256_file(args.census / name) for name in CENSUS_FILES},
        "tables": {
            **{f"{day}.csv": sha256_file(args.conn / f"{day}.csv") for day in DAYS},
            **{f"{day}_manifest.json": sha256_file(args.conn / f"{day}_manifest.json")
               for day in DAYS},
            **{f"labelled/{day}.csv": sha256_file(args.conn / "labelled" / f"{day}.csv")
               for day in DAYS},
        },
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
    current = bindings(args)
    labelled = json.loads((args.census / "label.json").read_text())
    for day in DAYS:
        manifest = json.loads((args.conn / f"{day}_manifest.json").read_text())
        entry, tables = labelled["tables"][day], current["tables"]
        if not (manifest["table_sha256"] == entry["source_sha256"] == tables[f"{day}.csv"]
                and entry["table_sha256"] == tables[f"labelled/{day}.csv"]):
            raise ValueError(f"{day}: tables differ from the conversion manifest or label census")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(path, {
        "status": "registered_before_featurization_and_any_score",
        "registered_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        **current,
    })
    print(json.dumps({"registered": str(path)}))


def local_hour(minute: int) -> int:
    return (minute * 60 + CAPTURE_OFFSET_SECONDS) // 3600 % 24


def featurize(args: argparse.Namespace) -> None:
    views_dir = args.output / "views"
    report_path = args.output / "featurize_report.json"
    if report_path.exists() or views_dir.exists():
        raise FileExistsError("Views already exist")
    protocol, _ = check_registration(args)
    roles = protocol["roles"]
    started = time.perf_counter()
    groups, settings = _settings()
    records = read_ton_records([args.conn / "labelled" / f"{day}.csv" for day in DAYS])
    support = json.loads((args.census / "support.json").read_text())
    checks: dict[str, bool] = {"ingestion_equals_census": records.report == support["ingestion"]}
    windows = records.windows()
    by_capture: dict[str, list[int]] = defaultdict(list)
    for minute in sorted(windows):
        by_capture[capture_of(minute * WINDOW_NS)].append(minute)
    observed = {
        capture: {
            "windows": len(minutes),
            "natural_normal_windows": sum(not records.attack[windows[m]].any() for m in minutes),
            "attack_windows": sum(bool(records.attack[windows[m]].any()) for m in minutes),
        }
        for capture, minutes in by_capture.items()
    }
    checks["window_counts_equal_census"] = observed == {
        capture: {key: item[key] for key in ("windows", "natural_normal_windows", "attack_windows")}
        for capture, item in support["captures"].items()
    }
    if not all(checks.values()):
        raise RuntimeError(f"Integrity check failed before featurization: {checks}")
    endpoints = records.endpoints

    def build(rows: np.ndarray, view: str, extra: dict[str, Any]) -> tuple[GraphSample, Any]:
        graph, flows, _ = records.view(rows, groups, **settings)
        attack_rows = rows[records.attack[rows]]
        touching: Counter[str] = Counter()
        for row in rows:
            touching[endpoints[records.sources[row]]] += 1
            touching[endpoints[records.destinations[row]]] += 1
        capture = capture_of(int(records.timestamps[rows[0]]))
        graph.metadata.update(
            view=view,
            capture=capture,
            session_key=session_key(capture),
            minute=int(records.timestamps[rows[0]] // WINDOW_NS),
            records=len(rows),
            attack_records=len(attack_rows),
            attack_endpoints=sorted(
                {endpoints[i] for i in records.sources[attack_rows]}
                | {endpoints[i] for i in records.destinations[attack_rows]}
            ),
            attack_types=sorted({records.type_names[t] for t in records.types[attack_rows]}),
            endpoint_records=[touching[key] for key in graph.metadata["endpoint_keys"]],
            **extra,
        )
        return graph, flows

    built: dict[str, tuple[list[GraphSample], list[torch.Tensor]]] = {v: ([], []) for v in VIEWS}
    source_minutes = by_capture[roles["source_capture"]]
    for role, members in zip(
        SOURCE_ROLES, _split_ordered_items(source_minutes, SOURCE_FRACTIONS), strict=True
    ):
        for minute in members:
            graph, flows = build(windows[minute], "source", {"role": role})
            built["source"][0].append(graph)
            built["source"][1].append(flows)
    for capture in roles["target_captures"]:
        for minute in by_capture[capture]:
            rows = windows[minute]
            attack = records.attack[rows]
            graph, flows = build(rows, "window", {"natural_normal": not attack.any()})
            built["window"][0].append(graph)
            built["window"][1].append(flows)
            if attack.any() and not attack.all():
                graph, flows = build(rows[~attack], "background", {"natural_normal": False})
                built["background"][0].append(graph)
                built["background"][1].append(flows)
        print(json.dumps({"featurized": capture, "seconds": round(time.perf_counter() - started)}),
              flush=True)

    checks["source_windows_all_normal"] = all(g.label == 0 for g in built["source"][0])
    checks["background_samples_label_0_only"] = all(
        g.label == 0 and g.metadata["attack_records"] == 0 for g in built["background"][0])
    checks["attack_windows_have_attack_records"] = all(
        (g.label == 1) == (g.metadata["attack_records"] > 0) == (not g.metadata["natural_normal"])
        for g in built["window"][0])
    checks["flows_finite"] = all(
        bool(torch.isfinite(f).all()) for view in VIEWS for f in built[view][1])
    checks["endpoint_records_aligned"] = all(
        len(g.metadata["endpoint_records"]) == g.node_features.shape[0]
        for view in VIEWS for g in built[view][0])
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
    write_json(report_path, {
        "status": "featurized_after_registration_no_scores",
        "ingestion": records.report,
        "integrity_checks": checks,
        "counts": {view: dict(Counter(g.metadata["capture"] for g in built[view][0]))
                   for view in VIEWS},
        "source_roles": dict(Counter(g.metadata["role"] for g in built["source"][0])),
        "view_sha256": hashes,
        "seconds": round(time.perf_counter() - started),
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    print(json.dumps({"featurized": {view: len(built[view][0]) for view in VIEWS}}), flush=True)


def load_views(args: argparse.Namespace, groups) -> dict[str, tuple[list[GraphSample], list]]:
    views_dir = args.output / "views"
    report = json.loads((args.output / "featurize_report.json").read_text())
    for name, digest in report["view_sha256"].items():
        if sha256_file(views_dir / name) != digest:
            raise ValueError(f"Featurized view changed: {name}")
    return {
        view: (load_graph_samples(views_dir / f"{view}_graphs.pt", groups),
               torch.load(views_dir / f"{view}_flows.pt", weights_only=True))
        for view in VIEWS
    }


def recent_thresholds(scores: np.ndarray, minutes: np.ndarray, vetted: np.ndarray,
                      captures: np.ndarray, memory: int, delay: int) -> np.ndarray:
    """Per-window conformal thresholds from the last `memory` vetted windows of the same
    capture that start more than `delay` minutes before the window; nan if fewer."""

    thresholds = np.full(len(scores), np.nan)
    for capture in np.unique(captures):
        own = np.flatnonzero(captures == capture)
        own = own[np.argsort(minutes[own], kind="stable")]
        pool = own[vetted[own]]
        pool_minutes, pool_scores = minutes[pool], scores[pool]
        for index in own:
            count = int(np.searchsorted(pool_minutes, minutes[index] - delay, side="left"))
            if count >= memory:
                thresholds[index] = conformal_threshold(pool_scores[count - memory:count], QUANTILE)
    return thresholds


def summarize(scores: np.ndarray, thresholds: np.ndarray, positive: np.ndarray,
              captures: np.ndarray, minimum: int) -> dict[str, Any]:
    supported = np.isfinite(thresholds)
    alerts = supported & (scores > np.where(supported, thresholds, np.inf))
    negative_mask, positive_mask = supported & ~positive, supported & positive
    fp, tp = int((alerts & negative_mask).sum()), int((alerts & positive_mask).sum())
    negatives, positives = int(negative_mask.sum()), int(positive_mask.sum())
    result: dict[str, Any] = {
        "negatives": negatives, "positives": positives, "fp": fp, "tp": tp,
        "fpr": fp / negatives if negatives else None,
        "tpr": tp / positives if positives else None,
        "binomial_p_fpr_leq_target": binomial_upper_tail(fp, negatives, TARGET),
        "auroc": (auroc(scores[negative_mask], scores[positive_mask])
                  if negatives and positives else None),
        "per_capture": {},
    }
    if negatives and positives:
        ordered = np.sort(scores[negative_mask])
        oracle = ordered[math.ceil(0.95 * negatives) - 1]
        result["oracle_tpr_at_eval_fpr_5pct"] = float(np.mean(scores[positive_mask] > oracle))
    for capture in np.unique(captures):
        mine = captures == capture
        n, p = int((negative_mask & mine).sum()), int((positive_mask & mine).sum())
        result["per_capture"][str(capture)] = {
            "negatives": n, "positives": p,
            "fp": int((alerts & negative_mask & mine).sum()),
            "tp": int((alerts & positive_mask & mine).sum()),
            "fpr": float((alerts & negative_mask & mine).sum() / n) if n else None,
            "tpr": float((alerts & positive_mask & mine).sum() / p) if p else None,
        }
    eligible = [c for c in result["per_capture"].values() if c["negatives"] >= minimum]
    result["safety_holds"] = bool(result["binomial_p_fpr_leq_target"] >= 0.05 and all(
        c["fpr"] <= 0.10 for c in eligible))
    result["_alerts"] = alerts
    return result


def block_statistic(numerator: np.ndarray, denominator: np.ndarray, blocks: np.ndarray,
                    universe: list[Any], draws: np.ndarray) -> dict[str, Any]:
    """Ratio of block sums with a block bootstrap (95% interval)."""

    index = {block: i for i, block in enumerate(universe)}
    top, bottom = np.zeros(len(universe)), np.zeros(len(universe))
    for value, weight, block in zip(numerator, denominator, blocks, strict=True):
        if block in index:
            top[index[block]] += value
            bottom[index[block]] += weight
    samples = [float(d @ top / (d @ bottom)) for d in draws if d @ bottom]
    return {
        "point": float(top.sum() / bottom.sum()) if bottom.sum() else None,
        "lower": float(np.quantile(samples, 0.025)) if samples else None,
        "upper": float(np.quantile(samples, 0.975)) if samples else None,
        "resamples_used": len(samples),
    }


def verdict(safe: bool, difference: dict[str, Any]) -> str:
    point, lower = difference["point"], difference["lower"]
    if safe and point is not None and point > 0:
        return "supported" if lower is not None and lower > 0 else "inconclusive"
    return "not_supported"


def gated_thresholds(scores: np.ndarray, minutes: np.ndarray, vetted: np.ndarray,
                     captures: np.ndarray, memory: int) -> tuple[np.ndarray, bool]:
    """Label-free alert-gated self-calibration seeded with the first `memory` vetted windows."""

    thresholds, monotone = np.full(len(scores), np.nan), True
    for capture in np.unique(captures):
        own = np.flatnonzero(captures == capture)
        own = own[np.argsort(minutes[own], kind="stable")]
        seed = own[vetted[own]][:memory]
        if len(seed) < memory:
            continue
        buffer, previous = scores[seed].tolist(), math.inf
        for index in own:
            if minutes[index] <= minutes[seed[-1]]:
                continue
            threshold = conformal_threshold(buffer, QUANTILE)
            monotone &= threshold <= previous
            previous = thresholds[index] = threshold
            if scores[index] <= threshold:
                buffer = [*buffer, float(scores[index])][-memory:]
    return thresholds, bool(monotone)


def evaluate(args: argparse.Namespace) -> None:
    destination = args.output / "report.json"
    if destination.exists():
        raise FileExistsError(destination)
    protocol, registration = check_registration(args)
    started = time.perf_counter()
    groups, _ = _settings()
    views = load_views(args, groups)
    source_graphs = views["source"][0]
    role = np.asarray([g.metadata["role"] for g in source_graphs])
    scaler = FeatureStandardizer.fit([g for g, r in zip(source_graphs, role, strict=True)
                                      if r == "train"])

    def windows(view: str, members) -> TrafficWindows:
        graphs, flows = views[view]
        members = list(members)
        return TrafficWindows(scaler.transform_many([graphs[i] for i in members], groups),
                              [flows[i] for i in members])

    source = {r: windows("source", np.flatnonzero(role == r)) for r in SOURCE_ROLES}
    detector = RawFlowDetector.fit(groups, source["train"], source["selection"],
                                   source["calibration"], k=DEFAULT_K, flow_variant="all")
    graphs = views["window"][0]
    fused, channels, entities = detector.score(windows("window", range(len(graphs))))
    print(json.dumps({"scored_windows": len(graphs),
                      "seconds": round(time.perf_counter() - started)}), flush=True)

    captures = np.asarray([g.metadata["capture"] for g in graphs])
    minutes = np.asarray([g.metadata["minute"] for g in graphs], dtype=np.int64)
    positive = np.asarray([not g.metadata["natural_normal"] for g in graphs])
    vetted = ~positive
    blocks = np.asarray([f"{c} {local_hour(m):02d}" for c, m in zip(captures, minutes,
                                                                    strict=True)])
    source_cal = [source_graphs[i] for i in np.flatnonzero(role == "calibration")]
    record_cal = np.log1p([g.metadata["records"] for g in source_cal])
    scores = {
        "detector": fused,
        "record_count": np.log1p([g.metadata["records"] for g in graphs]),
        "endpoint_count": np.log1p([len(g.metadata["endpoint_keys"]) for g in graphs]),
        "raw_node": channels["raw_node"],
        "flow": channels["flow"],
    }
    scales = {"detector": float(np.quantile(detector.calibration_scores, 0.95)),
              "record_count": float(np.quantile(record_cal, 0.95))}
    scores["combined"] = np.maximum(scores["detector"] / scales["detector"],
                                    scores["record_count"] / scales["record_count"])
    minimum = int(protocol["metrics"]["safety_capture_minimum"])
    boot = protocol["metrics"]["bootstrap"]
    rng = np.random.default_rng(boot["seed"])

    def policy(name: str, memory: int = 20, delay: int = 0) -> np.ndarray:
        return recent_thresholds(scores[name], minutes, vetted, captures, memory, delay)

    primary = {name: policy(name) for name in (*SCORES, "raw_node", "flow")}
    summaries = {name: summarize(scores[name], primary[name], positive, captures, minimum)
                 for name in primary}
    universe = sorted(set(blocks[np.isfinite(primary["detector"])]))
    draws = rng.multinomial(len(universe), np.full(len(universe), 1 / len(universe)),
                            boot["resamples"])

    def tpr_difference(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> dict:
        chosen = mask & positive & np.isfinite(primary["detector"])
        return block_statistic(first[chosen].astype(float) - second[chosen].astype(float),
                               np.ones(int(chosen.sum())), blocks[chosen], universe, draws)

    alerts = {name: summary.pop("_alerts") for name, summary in summaries.items()}
    everything = np.ones(len(graphs), dtype=bool)
    h1 = summaries["detector"]
    h2 = tpr_difference(alerts["detector"], alerts["record_count"], everything)
    h2_verdict = verdict(h1["safety_holds"], h2)
    report: dict[str, Any] = {
        "status": protocol["protocol"]["status"],
        "protocol_sha256": next(iter(registration["protocol"].values())),
        "policy": "recent_vetted_m20_d0",
        "primary_scores": summaries,
        "primary": {
            "H1_safety_holds": h1["safety_holds"],
            "H1_binomial_p": h1["binomial_p_fpr_leq_target"],
            "H1_per_capture_fpr": {c: v["fpr"] for c, v in h1["per_capture"].items()},
            "H2_detector_minus_record_count_tpr": h2,
            "H2_beyond_volume": h2_verdict,
            "validated": bool(h1["safety_holds"] and h2_verdict == "supported"),
        },
    }

    # H3 and the staleness curve on windows supported at every delay.
    delays = protocol["secondary"]["H3_staleness"]["delays_minutes"]
    curve = {name: {d: policy(name, 20, d) for d in delays}
             for name in ("detector", "record_count", "combined")}
    common = np.logical_and.reduce([np.isfinite(curve["detector"][d]) for d in delays])

    def rate(name: str, delay: int, mask: np.ndarray) -> float | None:
        chosen = common & mask
        return (float(np.mean(scores[name][chosen] > curve[name][delay][chosen]))
                if chosen.any() else None)

    report["H3_staleness"] = {
        "common_negatives": int((common & ~positive).sum()),
        "common_positives": int((common & positive).sum()),
        "curve": {name: {str(d): {"fpr": rate(name, d, ~positive), "tpr": rate(name, d, positive)}
                         for d in delays} for name in curve},
    }
    last = max(delays)

    def false_alerts(name: str, delay: int) -> np.ndarray:
        chosen = common & ~positive
        return (scores[name][chosen] > curve[name][delay][chosen]).astype(float)

    did = (false_alerts("record_count", last) - false_alerts("record_count", 0)) - (
        false_alerts("detector", last) - false_alerts("detector", 0))
    chosen = common & ~positive
    h3 = block_statistic(did, np.ones(int(chosen.sum())), blocks[chosen], universe, draws)
    report["H3_staleness"]["record_minus_detector_fpr_increase"] = h3
    report["H3_staleness"]["supported"] = bool(h3["point"] is not None and h3["point"] > 0
                                               and h3["lower"] is not None and h3["lower"] > 0)

    # S1 attack types and volume strata.
    types = [set(g.metadata["attack_types"]) for g in graphs]
    supported = np.isfinite(primary["detector"])
    s1: dict[str, Any] = {"types": {}, "strata": {}}
    for kind in sorted(set().union(*types)):
        mask = np.asarray([kind in t for t in types]) & supported & positive
        s1["types"][kind] = {
            "positives": int(mask.sum()),
            **{f"{name}_tpr": float(alerts[name][mask].mean()) if mask.any() else None
               for name in ("detector", "record_count", "combined")},
            "median_records": float(np.median(np.expm1(scores["record_count"][mask])))
            if mask.any() else None,
        }
    strata = {"low_volume": {"web_brute_force", "web_xss", "web_sql_injection", "botnet"},
              "high_volume": {"portscan", "ddos"}, "infiltration": {"infiltration"}}
    for stratum, members in strata.items():
        mask = np.asarray([bool(t & members) for t in types])
        s1["strata"][stratum] = {
            "positives": int((mask & supported & positive).sum()),
            "detector_minus_record_count_tpr": tpr_difference(
                alerts["detector"], alerts["record_count"], mask),
        }
    report["S1_attack_types"] = s1

    # S2 static Monday thresholds on every target window; source audit and test.
    static = {"detector": detector.thresholds[f"{TARGET:g}"],
              "record_count": conformal_threshold(record_cal, QUANTILE)}
    report["S2_static_source_thresholds"] = {
        name: {key: value for key, value in summarize(
            scores[name], np.full(len(graphs), threshold), positive, captures, minimum
        ).items() if key != "_alerts"}
        for name, threshold in static.items()
    }
    report["S2_static_source_thresholds"]["source_fpr"] = {
        r: float(np.mean(detector.score(source[r])[0] > static["detector"]))
        for r in ("audit", "test")
    }

    # S3 memory 60 and the combined score (already in primary_scores).
    report["S3_memory_60"] = {
        name: {key: value for key, value in summarize(
            scores[name], policy(name, 60, 0), positive, captures, minimum).items()
            if key != "_alerts"}
        for name in ("detector", "record_count")
    }

    # S4 label-free alert-gated self-calibration.
    report["S4_label_free_alert_gated"] = {}
    for name in ("detector", "record_count"):
        thresholds, monotone = gated_thresholds(scores[name], minutes, vetted, captures, 20)
        item = summarize(scores[name], thresholds, positive, captures, minimum)
        item.pop("_alerts")
        report["S4_label_free_alert_gated"][name] = {**item, "thresholds_non_increasing": monotone}

    # S5 background-contrast negatives against their window's primary threshold.
    background = views["background"][0]
    lookup = {(c, m): i for i, (c, m) in enumerate(zip(captures, minutes, strict=True))}
    position = np.asarray([lookup[(g.metadata["capture"], g.metadata["minute"])]
                           for g in background], dtype=np.int64)
    background_scores = {
        "detector": detector.score(windows("background", range(len(background))))[0]
        if background else np.zeros(0),
        "record_count": np.log1p([g.metadata["records"] for g in background]),
    }
    report["S5_background_contrast"] = {}
    for name, values in background_scores.items():
        threshold = primary[name][position] if len(position) else np.zeros(0)
        ok = np.isfinite(threshold)
        report["S5_background_contrast"][name] = {
            "samples": int(ok.sum()),
            "fpr": float(np.mean(values[ok] > threshold[ok])) if ok.any() else None,
        }

    # S6 channels (in primary_scores as raw_node and flow).
    report["S6_channels"] = {name: {k: summaries[name][k] for k in ("auroc", "tpr", "fpr")}
                             for name in ("raw_node", "flow")}

    # S7 entities within attack windows.
    distances, volumes, labels, per_window = [], [], [], {"distance": [], "records": []}
    for index in np.flatnonzero(positive):
        metadata = graphs[index].metadata
        involved = set(metadata["attack_endpoints"])
        own_labels = np.asarray([key in involved for key in metadata["endpoint_keys"]])
        own_distance = entities["raw_node"][index].numpy()
        own_records = np.asarray(metadata["endpoint_records"], dtype=np.float64)
        if len(own_distance) != len(own_labels):
            raise ValueError("Entity scores and endpoint keys are misaligned")
        labels.extend(own_labels.tolist())
        distances.extend(own_distance.tolist())
        volumes.extend(own_records.tolist())
        if own_labels.any() and not own_labels.all():
            per_window["distance"].append(auroc(own_distance[~own_labels],
                                                own_distance[own_labels]))
            per_window["records"].append(auroc(own_records[~own_labels], own_records[own_labels]))
    labels_array = np.asarray(labels, dtype=bool)
    report["S7_entities"] = {
        "attack_endpoints": int(labels_array.sum()),
        "other_endpoints": int((~labels_array).sum()),
        **{f"{name}_pooled_auroc": auroc(np.asarray(values)[~labels_array],
                                         np.asarray(values)[labels_array])
           for name, values in (("distance", distances), ("records", volumes))},
        **{f"{name}_mean_window_auroc": float(np.mean(values)) if values else None
           for name, values in per_window.items()},
        "windows_with_both_classes": len(per_window["distance"]),
    }
    report["seconds"] = round(time.perf_counter() - started)
    report["finished_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    json.dumps(report, allow_nan=False)  # fail before writing anything
    with (args.output / "scores.pt").open("xb") as stream:
        torch.save({
            "window": {name: torch.from_numpy(np.asarray(values, dtype=np.float64))
                       for name, values in scores.items()},
            "primary_thresholds": {name: torch.from_numpy(values)
                                   for name, values in primary.items()},
            "captures": captures.tolist(),
            "minutes": torch.from_numpy(minutes),
            "positive": torch.from_numpy(positive),
            "background_detector": torch.from_numpy(np.asarray(background_scores["detector"])),
            "monday_calibration_detector": torch.from_numpy(detector.calibration_scores),
        }, stream)
    write_json(destination, report)
    print(json.dumps({"primary": report["primary"],
                      "H3": report["H3_staleness"]["supported"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "featurize", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--conn", type=Path, default=CONN)
    parser.add_argument("--census", type=Path, default=CENSUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    {"register": register, "featurize": featurize, "evaluate": evaluate}[args.phase](args)


if __name__ == "__main__":
    main()
