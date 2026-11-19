"""Explore information lost by the v3b schema, with fixed feature ablations.

Uses the original role/window assignments. Protocol, service, flags, duration,
ports and byte counts are measured traffic fields; no host identity, date,
label, IDS alert or host-log field enters a feature vector. All variants are
reported. References are train-normal, fusion scales are select-normal, and
thresholds are calibration-normal. Test/audit are already consumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_flow_headroom import FEATURES, edge_table  # noqa: E402
from diagnose_raw_records import WINDOW_NS, read_rows  # noqa: E402
from investigate_detection import (  # noqa: E402
    DEFAULT_OUTPUT,
    PREPARED,
    ROLE_PATHS,
    ROOT,
    write_json,
)

from fedxmag.confirmation import sha256_file  # noqa: E402
from fedxmag.local_support import LocalSupport  # noqa: E402
from fedxmag.m1.detection import MahalanobisDetector, Standardizer, detection_summary  # noqa: E402
from fedxmag.m1.entity import aggregate, encode_entities, stratified_aurocs  # noqa: E402
from fedxmag.scoring import FPR_TARGETS, FrozenNodeDetector  # noqa: E402
from fedxmag.serialization import load_graph_samples  # noqa: E402

ROLES = {**ROLE_PATHS, "selection": PREPARED / "select_normal.pt"}
FEATURE_SETS = {
    "all_flow": list(range(len(FEATURES))),
    "without_protocol_service": [i for i in range(len(FEATURES)) if not 1 <= i <= 6],
    "flags_only": list(range(7, 12)),
}


def context_cache(samples: dict, output: Path) -> dict:
    path = output / "flow_context_cache.pt"
    raw = ROOT / "data/X-IIoTID/X-IIoTID dataset.csv"
    bindings = {
        "raw_sha256": sha256_file(raw),
        **{role: sha256_file(p) for role, p in ROLES.items()},
    }
    if path.exists():
        cache = torch.load(path, weights_only=True)
        if cache["bindings"] != bindings or cache["feature_names"] != list(FEATURES):
            raise ValueError("Flow cache sources changed")
        return cache["features"]
    desired = {
        (s.metadata["session_key"], s.metadata["window_start_ns"] // WINDOW_NS)
        for graphs in samples.values()
        for s in graphs
    }
    print(json.dumps({"phase": "reading_raw_context", "desired_windows": len(desired)}), flush=True)
    rows = read_rows(raw)
    keys = [
        (s, int(t // WINDOW_NS)) for s, t in zip(rows["sessions"], rows["timestamps"], strict=True)
    ]
    mask = np.asarray([key in desired for key in keys])
    tables = edge_table(rows, mask)
    features = {
        role: [
            torch.from_numpy(
                np.stack(
                    tables[(s.metadata["session_key"], s.metadata["window_start_ns"] // WINDOW_NS)]
                )
            )
            for s in graphs
        ]
        for role, graphs in samples.items()
    }
    # Verify that raw rows assigned to the normal-only fit roles contain no attacks.
    for role in ("train", "selection", "calibration"):
        allowed = {
            (s.metadata["session_key"], s.metadata["window_start_ns"] // WINDOW_NS)
            for s in samples[role]
        }
        if any(
            bool(label) and key in allowed for key, label in zip(keys, rows["labels"], strict=True)
        ):
            raise ValueError(f"Attack row in normal-only context role: {role}")
    with path.open("xb") as stream:
        torch.save(
            {"bindings": bindings, "feature_names": list(FEATURES), "features": features}, stream
        )
    print(
        json.dumps(
            {"phase": "context_ready", "reference_edges": sum(len(t) for t in features["train"])}
        ),
        flush=True,
    )
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "context_study.json").exists():
        raise FileExistsError("Context study output already exists")
    torch.set_num_threads(4)
    detector = FrozenNodeDetector.load(ROOT / "output/entity_detector/detector_seed_17.pt")
    samples = {role: load_graph_samples(path, detector.groups) for role, path in ROLES.items()}
    features = context_cache(samples, output)
    # These feature sets and estimators were fixed before inspecting their outputs.
    graph_scores = {}
    for name, columns in FEATURE_SETS.items():
        reference = torch.cat(features["train"])[:, columns]
        lw = MahalanobisDetector.fit(reference, "ledoit_wolf", standardize=True)
        scaler = Standardizer.fit(reference)
        local = LocalSupport.fit(scaler(reference), k=20)
        for estimator in ("lw", "knn"):
            graph_scores[f"{name}_{estimator}"] = {}
        for role, rows in features.items():
            if role == "train":
                continue
            values = torch.cat(rows)[:, columns]
            lengths = [len(t) for t in rows]
            for estimator, scores in (
                ("lw", lw.score(values)),
                ("knn", local.score(scaler(values))["knn"]),
            ):
                graph_scores[f"{name}_{estimator}"][role] = aggregate(
                    scores.split(lengths), "top3_mean"
                )
        print(json.dumps({"phase": "scored_context", "features": name}), flush=True)

    # A priori fusion: max of two score channels normalized by their 95th
    # percentile on select-normal. The independent calibration role sets thresholds.
    for seed in (17, 29, 41, 53, 67):
        detector = FrozenNodeDetector.load(ROOT / f"output/entity_detector/detector_seed_{seed}.pt")
        path = output / f"mechanism_scores_seed_{seed}.pt"
        if path.exists():
            scores = torch.load(path, weights_only=True)["graph"]["mahalanobis|top3_mean"]
            node = {role: value.numpy() for role, value in scores.items()}
        else:
            node = {
                role: detector.score(graphs)[0]
                for role, graphs in samples.items()
                if role not in ("train", "selection")
            }
        states = encode_entities(detector.model, samples["selection"], detector.groups)
        node["selection"] = detector.scorer.graph_scores(states, "top3_mean")
        graph_scores[f"node_lw_seed_{seed}"] = node
        for context_name in ("all_flow_lw", "all_flow_knn"):
            flow = graph_scores[context_name]
            node_scale = max(float(np.quantile(node["selection"], 0.95)), 1e-12)
            flow_scale = max(float(np.quantile(flow["selection"], 0.95)), 1e-12)
            graph_scores[f"node_plus_{context_name}_seed_{seed}"] = {
                role: np.maximum(node[role] / node_scale, flow[role] / flow_scale) for role in node
            }
        print(json.dumps({"phase": "fused_context", "seed": seed}), flush=True)

    context = {}
    for band in ("test", "audit"):
        path = ROOT / (
            "output/diagnosis/test_band_windows.json"
            if band == "test"
            else "data/processed/xiiotid_stage1_v3b_audit_band/audit_attack_windows.json"
        )
        windows = json.loads(path.read_text())
        lookup = {(w["session"], w["window_start_ns"]): w for w in windows}
        normal, attack = samples[f"{band}_normal"], samples[f"{band}_attack"]
        context[band] = {
            "normal_nodes": np.asarray([s.node_features.shape[0] for s in normal]),
            "attack_nodes": np.asarray([s.node_features.shape[0] for s in attack]),
            "normal_dates": np.asarray([s.metadata["session_key"] for s in normal]),
            "attack_dates": np.asarray([s.metadata["session_key"] for s in attack]),
            "attack_fraction": np.asarray(
                [
                    lookup[(s.metadata["session_key"], s.metadata["window_start_ns"])][
                        "attack_fraction"
                    ]
                    for s in attack
                ]
            ),
        }
    report = {
        "status": "exploratory_feature_information_study",
        "feature_names": list(FEATURES),
        "feature_sets": {name: [FEATURES[i] for i in cols] for name, cols in FEATURE_SETS.items()},
        "fusion": "max of select-normal q95-normalized channels",
        "methods": {},
    }
    for name, scores in graph_scores.items():
        report["methods"][name] = {}
        for band in ("test", "audit"):
            normal, attack = scores[f"{band}_normal"], scores[f"{band}_attack"]
            summary = detection_summary(scores["calibration"], normal, attack, FPR_TARGETS)
            summary.update(stratified_aurocs(normal, attack, **context[band]))
            report["methods"][name][band] = summary
        print(
            json.dumps(
                {
                    "method": name,
                    **{
                        band: {
                            "auroc": round(report["methods"][name][band]["auroc"], 4),
                            "within_date": round(
                                report["methods"][name][band]["within_date_auroc"], 4
                            ),
                            **{
                                m: round(
                                    report["methods"][name][band]["at_calibrated_fpr"]["0.05"][m], 4
                                )
                                for m in ("tpr", "fpr")
                            },
                        }
                        for band in ("test", "audit")
                    },
                }
            ),
            flush=True,
        )
    write_json(output / "context_study.json", report)
    with (output / "context_scores.pt").open("xb") as stream:
        torch.save(
            {
                name: {role: torch.from_numpy(values) for role, values in roles.items()}
                for name, roles in graph_scores.items()
            },
            stream,
        )


if __name__ == "__main__":
    main()
