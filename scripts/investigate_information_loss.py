"""Feature ablations and exact feature collisions, all explicitly exploratory."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from investigate_detection import DEFAULT_OUTPUT, ROLE_PATHS, ROOT, write_json

from fedxmag.local_support import LocalSupport
from fedxmag.m1.detection import Standardizer, detection_summary
from fedxmag.m1.entity import aggregate, raw_entity_features, stratified_aurocs
from fedxmag.scoring import FPR_TARGETS, FrozenNodeDetector
from fedxmag.serialization import load_graph_samples


def fingerprint(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


def graph_fingerprint(sample) -> str:
    return fingerprint(
        [
            sample.node_features,
            sample.node_available,
            sample.edge_features,
            sample.edge_available,
            sample.edge_index,
        ]
    )


def main() -> None:
    output = DEFAULT_OUTPUT
    if (output / "information_loss.json").exists():
        raise FileExistsError("Information-loss report exists")
    torch.set_num_threads(4)
    detector = FrozenNodeDetector.load(ROOT / "output/entity_detector/detector_seed_17.pt")
    samples = {role: load_graph_samples(path, detector.groups) for role, path in ROLE_PATHS.items()}
    cache = torch.load(output / "flow_context_cache.pt", weights_only=True)
    flow = cache["features"]
    variants = {
        "flow_activity": [0, 12, 15, 16],  # counts, duration, source/destination byte volume
        "flow_activity_flags": [0, *range(7, 13), 15, 16],
        "flow_protocol_service": list(range(1, 7)),
    }
    graph_scores = {}
    for name, columns in variants.items():
        reference = torch.cat(flow["train"])[:, columns]
        scaler = Standardizer.fit(reference)
        local = LocalSupport.fit(scaler(reference), k=20)
        graph_scores[name] = {}
        for role, rows in flow.items():
            if role in ("train", "selection"):
                continue
            scores = local.score(scaler(torch.cat(rows)[:, columns]))["knn"]
            graph_scores[name][role] = aggregate(scores.split([len(t) for t in rows]), "top3_mean")
    raw = {role: raw_entity_features(graphs) for role, graphs in samples.items()}
    for kind in ("node", "edge"):
        local = LocalSupport.fit(raw["train"].stacked(kind), k=20)
        name = f"v3b_raw_{kind}_knn"
        graph_scores[name] = {}
        for role, states in raw.items():
            if role == "train":
                continue
            rows = getattr(states, kind)
            scores = local.score(states.stacked(kind))["knn"]
            graph_scores[name][role] = aggregate(scores.split([len(t) for t in rows]), "top3_mean")

    report = {
        "status": "exploratory_feature_ablation",
        "methods": {},
        "collisions": {},
        "features": {
            name: [cache["feature_names"][i] for i in cols] for name, cols in variants.items()
        },
    }
    train_hashes = {graph_fingerprint(sample) for sample in samples["train"]}
    raw_train_rows = {
        kind: {fingerprint([row]) for row in raw["train"].stacked(kind)}
        for kind in ("node", "edge")
    }
    for band in ("test", "audit"):
        info_path = ROOT / (
            "output/diagnosis/test_band_windows.json"
            if band == "test"
            else "data/processed/xiiotid_stage1_v3b_audit_band/audit_attack_windows.json"
        )
        lookup = {
            (w["session"], w["window_start_ns"]): w for w in json.loads(info_path.read_text())
        }
        normal, attack = samples[f"{band}_normal"], samples[f"{band}_attack"]
        info = [lookup[(s.metadata["session_key"], s.metadata["window_start_ns"])] for s in attack]
        context = {
            "normal_nodes": np.asarray([s.node_features.shape[0] for s in normal]),
            "attack_nodes": np.asarray([s.node_features.shape[0] for s in attack]),
            "normal_dates": np.asarray([s.metadata["session_key"] for s in normal]),
            "attack_dates": np.asarray([s.metadata["session_key"] for s in attack]),
            "attack_fraction": np.asarray([w["attack_fraction"] for w in info]),
        }
        collided = sum(graph_fingerprint(s) in train_hashes for s in attack)
        report["collisions"][band] = {
            "exact_attack_graph_matches_to_train_normal": collided,
            "attack_windows": len(attack),
        }
        for kind in ("node", "edge"):
            matched, total = 0, 0
            for sample, state, window in zip(
                attack, getattr(raw[f"{band}_attack"], kind), info, strict=True
            ):
                if kind == "node":
                    involved = set(window["attack_endpoint_keys"])
                    keys = sample.metadata["endpoint_keys"]
                else:
                    pairs = {tuple(pair) for pair in window["attack_pair_keys"]}
                    involved = pairs | {(b, a) for a, b in pairs}
                    keys = [tuple(pair) for pair in sample.metadata["edge_keys"]]
                for key, row in zip(keys, state, strict=True):
                    if key in involved:
                        total += 1
                        matched += int(fingerprint([row]) in raw_train_rows[kind])
            report["collisions"][band][f"attack_{kind}_exact_feature_match_fraction"] = (
                matched / total
            )
        for name, scores in graph_scores.items():
            result = detection_summary(
                scores["calibration"],
                scores[f"{band}_normal"],
                scores[f"{band}_attack"],
                FPR_TARGETS,
            )
            result.update(
                stratified_aurocs(scores[f"{band}_normal"], scores[f"{band}_attack"], **context)
            )
            report["methods"].setdefault(name, {})[band] = result
            print(
                json.dumps(
                    {
                        "method": name,
                        "band": band,
                        "auroc": result["auroc"],
                        "within_date": result["within_date_auroc"],
                        "rates": result["at_calibrated_fpr"]["0.05"],
                    }
                ),
                flush=True,
            )
    write_json(output / "information_loss.json", report)
    with (output / "information_loss_scores.pt").open("xb") as stream:
        torch.save(
            {
                name: {role: torch.from_numpy(v) for role, v in scores.items()}
                for name, scores in graph_scores.items()
            },
            stream,
        )
    print(json.dumps(report["collisions"]), flush=True)


if __name__ == "__main__":
    main()
