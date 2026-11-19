"""Post-hoc trace of the TON_IoT false-positive overshoot (exploratory; TON_IoT is consumed).

Rebuilds the registered control detectors from the featurized views, checks that they
reproduce the archived scores exactly, and writes one row per calibration sample,
evaluation negative, and evaluation positive. Each row holds the sample's size, its
time position, its channel scores, and properties of its raw records: connection
states, records touching endpoints attacked in the same window, and endpoints absent
from the reference. Nothing here changes the registered verdict.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from run_entity_detector import write_json
from run_ton_iot_validation import (
    DATA,
    ENVIRONMENT,
    OUTPUT,
    PROTOCOL,
    SOURCE_ROLES,
    _settings,
    commissioning_split,
    load_views,
    tables,
)

from fedxmag.flow_context import FLOW_FEATURES, WINDOW_NS, flow_table
from fedxmag.fusion_detector import RawFlowDetector, TrafficWindows
from fedxmag.preprocessing.scaling import FeatureStandardizer
from fedxmag.ton_iot import read_ton_records

DIAGNOSIS = Path(__file__).resolve().parents[1] / "output/ton_iot_fpr_diagnosis"
STATES = ("SF", "S0", "REJ", "S1", "S3", "OTH", "SHR", "RSTOS0", "RSTO")


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = pd.Series(left).rank().to_numpy()
    b = pd.Series(right).rank().to_numpy()
    a, b = a - a.mean(), b - b.mean()
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


def build(args: argparse.Namespace) -> dict[str, Any]:
    groups, _ = _settings()
    views = load_views(args, groups)
    source_graphs, _ = views["source"]
    raw_role = np.asarray([g.metadata["role"] for g in source_graphs])
    scaler = FeatureStandardizer.fit([g for g, r in zip(source_graphs, raw_role, strict=True)
                                      if r == "train"])

    def windows(view: str, members) -> TrafficWindows:
        graphs, flows = views[view]
        return TrafficWindows(scaler.transform_many([graphs[i] for i in members], groups),
                              [flows[i] for i in members])

    source = {r: windows("source", np.flatnonzero(raw_role == r).tolist()) for r in SOURCE_ROLES}
    background, attack = views["background"][0], views["attack"][0]
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    split, _ = commissioning_split(background, attack, list(protocol["roles"]["target_captures"]))
    target = {
        "reference": windows("background", split["reference"]),
        "calibration": windows("background", split["calibration"]),
        "negative": windows("background", split["negative"]),
        "positive": windows("attack", split["positive"]),
    }
    fitted = RawFlowDetector.fit(groups, source["train"], source["selection"],
                                 source["calibration"], flow_variant="all")
    commissioned = fitted.commission(target["reference"], target["calibration"],
                                     environment=ENVIRONMENT)
    return {"views": views, "split": split, "target": target, "source": source,
            "source_detector": fitted, "commissioned": commissioned, "groups": groups,
            "scaler": scaler, "windows": windows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--diagnosis", type=Path, default=DIAGNOSIS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    table_path, summary_path = args.diagnosis / "samples.csv", args.diagnosis / "summary.json"
    if table_path.exists() or summary_path.exists():
        raise FileExistsError(args.diagnosis)
    state = build(args)
    split, target = state["split"], state["target"]
    background, attack = state["views"]["background"][0], state["views"]["attack"][0]
    archived = torch.load(args.output / "scores.pt", weights_only=True)["all"]
    detectors = {"recalibrated": state["source_detector"], "commissioned": state["commissioned"]}
    scored = {
        name: {role: detector.score(target[role]) for role in ("calibration", "negative",
                                                                "positive")}
        for name, detector in detectors.items()
    }
    replay = all(
        np.array_equal(scored[name][role][0], archived[name][role].numpy())
        for name in detectors for role in ("negative", "positive")
    ) and np.array_equal(scored["commissioned"]["calibration"][0],
                         archived["commissioned"]["calibration"].numpy())
    if not replay:
        raise RuntimeError("Rebuilt detectors do not reproduce the archived scores")
    arms = json.loads((args.output / "report.json").read_text())["variants"]["all"]["arms"]
    thresholds = {name: arms[name]["thresholds"]["0.05"] for name in detectors}
    if thresholds["commissioned"] != state["commissioned"].thresholds["0.05"]:
        raise RuntimeError("Rebuilt commissioned threshold differs from the report")

    print(json.dumps({"replay": replay, "reading_records": True}), flush=True)
    records = read_ton_records(tables(args.data))
    rows_of = records.windows()
    state_code = {name: i for i, name in enumerate(records.state_names)}
    reference_endpoints = {
        key for graphs in (state["source"]["train"].graphs, target["reference"].graphs)
        for g in graphs for key in g.metadata["endpoint_keys"]
    }
    capture_bounds: dict[str, tuple[int, int]] = {}
    for g in background + attack:
        lo, hi = capture_bounds.get(g.metadata["capture"], (10**20, -1))
        start = g.metadata["window_start_ns"]
        capture_bounds[g.metadata["capture"]] = (min(lo, start), max(hi, start))

    table = []
    entities = scored["commissioned"]
    for role, view, members in (("calibration", background, split["calibration"]),
                                ("negative", background, split["negative"]),
                                ("positive", attack, split["positive"])):
        fused, channels, ents = entities[role]
        recal_fused = scored["recalibrated"][role][0]
        for position, index in enumerate(members):
            graph = view[index]
            meta = graph.metadata
            minute = meta["window_start_ns"] // WINDOW_NS
            rows = rows_of[minute]
            label_rows = rows[records.attack[rows]]
            victims = ({records.endpoints[i] for i in records.sources[label_rows]}
                       | {records.endpoints[i] for i in records.destinations[label_rows]})
            own = rows if role == "positive" else rows[~records.attack[rows]]
            sources = [records.endpoints[i] for i in records.sources[own]]
            destinations = [records.endpoints[i] for i in records.destinations[own]]
            touching = np.asarray([s in victims or d in victims
                                   for s, d in zip(sources, destinations, strict=True)])
            states = Counter(records.states[own].tolist())
            lo, hi = capture_bounds[meta["capture"]]
            node_scores = ents["raw_node"][position]
            flow_scores = ents["flow"][position]
            top_node = meta["endpoint_keys"][int(torch.argmax(node_scores))]
            flow_keys = next(iter(flow_table(records.flow_records(own)).values()))[1]
            top_flow = flow_keys[int(torch.argmax(flow_scores))]
            endpoints = set(meta["endpoint_keys"])
            table.append({
                "role": role,
                "capture": meta["capture"],
                "minute": int(minute),
                "time_position": (meta["window_start_ns"] - lo) / max(hi - lo, 1),
                "records": len(own),
                "nodes": len(meta["endpoint_keys"]),
                "flows": len(flow_keys),
                "natural_normal": bool(meta.get("natural_normal", False)),
                "score": float(fused[position]),
                "recal_score": float(recal_fused[position]),
                "raw_node": float(channels["raw_node"][position]
                                  / state["commissioned"].scales["raw_node"]),
                "flow": float(channels["flow"][position] / state["commissioned"].scales["flow"]),
                "window_attack_records": len(label_rows),
                "victim_touching_share": float(touching.mean()) if len(touching) else 0.0,
                "new_endpoint_share": len(endpoints - reference_endpoints) / len(endpoints),
                "top_node_is_victim": top_node in victims,
                "top_node_is_new": top_node not in reference_endpoints,
                "top_flow_touches_victim": top_flow[0] in victims or top_flow[1] in victims,
                **{f"state_{name}": states.get(state_code.get(name, -1), 0) / max(len(own), 1)
                   for name in STATES},
            })
    frame = pd.DataFrame(table)
    for name, threshold in thresholds.items():
        column = "score" if name == "commissioned" else "recal_score"
        frame[f"alert_{name}"] = frame[column] > threshold

    summary: dict[str, Any] = {
        "status": "post_hoc_exploratory_on_consumed_ton_iot",
        "replay_of_archived_scores": replay,
        "thresholds_5pct": thresholds,
        "flow_features": list(FLOW_FEATURES),
    }
    negative = frame[frame.role == "negative"]
    calibration = frame[frame.role == "calibration"]
    positive = frame[frame.role == "positive"]
    summary["spearman_score_vs_log_records"] = {
        role: spearman(part.score.to_numpy(), np.log1p(part.records.to_numpy()))
        for role, part in (("calibration", calibration), ("negative", negative),
                           ("positive", positive))
    }
    summary["spearman_score_vs_nodes"] = {
        role: spearman(part.score.to_numpy(), part.nodes.to_numpy())
        for role, part in (("calibration", calibration), ("negative", negative))
    }
    edges = np.quantile(np.log1p(calibration.records), [0.2, 0.4, 0.6, 0.8])
    bins = {}
    for role, part in (("calibration", calibration), ("negative", negative)):
        codes = np.digitize(np.log1p(part.records), edges)
        bins[role] = {
            str(code): {
                "count": int((codes == code).sum()),
                "commissioned_alert_rate": float(part.alert_commissioned[codes == code].mean()),
                "recalibrated_alert_rate": float(part.alert_recalibrated[codes == code].mean()),
            }
            for code in range(5)
        }
    summary["alert_rate_by_calibration_record_quintile"] = {
        "log_record_edges": edges.tolist(), **bins
    }
    false_positive = negative[negative.alert_commissioned]
    true_negative = negative[~negative.alert_commissioned]
    columns = ["records", "nodes", "flows", "time_position", "victim_touching_share",
               "new_endpoint_share", "window_attack_records",
               *(f"state_{name}" for name in STATES)]
    summary["false_positive_vs_true_negative_medians"] = {
        column: [float(false_positive[column].median()), float(true_negative[column].median()),
                 float(calibration[column].median())]
        for column in columns
    }
    summary["false_positive_rates_of_flags"] = {
        flag: {"false_positive": float(false_positive[flag].mean()),
               "true_negative": float(true_negative[flag].mean()),
               "calibration": float(calibration[flag].mean())}
        for flag in ("top_node_is_victim", "top_node_is_new", "top_flow_touches_victim")
    }
    summary["dominant_channel_in_false_positives"] = float(
        (false_positive.flow >= false_positive.raw_node).mean()
    )
    touching_bins = [-1, 0, 0.1, 0.5, 1.0]
    codes = np.digitize(negative.victim_touching_share, touching_bins[1:], right=True)
    summary["alert_rate_by_victim_touching_share"] = {
        label: {"count": int((codes == code).sum()),
                "commissioned": float(negative.alert_commissioned[codes == code].mean())
                if (codes == code).any() else None}
        for code, label in enumerate(("0", "(0,0.1]", "(0.1,0.5]", "(0.5,1]"))
    }
    summary["finished_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    args.diagnosis.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_path, index=False)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
