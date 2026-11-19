"""Exploratory mechanism tests on consumed X-IIoTID evaluation bands.

All references use train-normal only; thresholds use calibration-normal only.
No checkpoint is trained or selected. Fixed alternatives distinguish global
ellipsoidal scoring, local density, aggregation, and feature availability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fedxmag.confirmation import sha256_file
from fedxmag.local_support import LocalSupport
from fedxmag.m1.detection import detection_summary
from fedxmag.m1.entity import aggregate, encode_entities, stratified_aurocs
from fedxmag.scoring import FPR_TARGETS, FrozenNodeDetector
from fedxmag.serialization import load_graph_samples

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data/processed/xiiotid_stage1_v3b_development"
DEFAULT_OUTPUT = ROOT / "output/root_cause_20261026"
ROLE_PATHS = {
    "train": PREPARED / "train_normal.pt",
    "calibration": PREPARED / "calibration_normal.pt",
    "test_normal": PREPARED / "test_normal.pt",
    "test_attack": PREPARED / "test_attack.pt",
    "audit_normal": PREPARED / "audit_normal.pt",
    "audit_attack": ROOT / "data/processed/xiiotid_stage1_v3b_audit_band/audit_attack.pt",
}


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def quantiles(values: np.ndarray) -> dict[str, float | None]:
    return {
        str(q): float(np.quantile(values, q)) if values.size else None
        for q in (0.1, 0.5, 0.9, 0.95, 0.99)
    }


def run(seed: int, output: Path) -> None:
    report_path = output / f"mechanisms_seed_{seed}.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    output.mkdir(parents=True, exist_ok=True)
    artifact = ROOT / f"output/entity_detector/detector_seed_{seed}.pt"
    detector = FrozenNodeDetector.load(artifact)
    samples = {role: load_graph_samples(path, detector.groups) for role, path in ROLE_PATHS.items()}
    cache_path = output / f"states_seed_{seed}.pt"
    fingerprints = {
        "detector": sha256_file(artifact),
        **{role: sha256_file(path) for role, path in ROLE_PATHS.items()},
    }
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=True)
        if cache["fingerprints"] != fingerprints:
            raise ValueError("State cache inputs changed")
        states = cache["states"]
    else:
        states = {
            role: encode_entities(detector.model, graphs, detector.groups).node
            for role, graphs in samples.items()
        }
        with cache_path.open("xb") as stream:
            torch.save({"fingerprints": fingerprints, "states": states}, stream)
    print(json.dumps({"seed": seed, "phase": "states_ready"}), flush=True)
    windows = {
        "test": json.loads((ROOT / "output/diagnosis/test_band_windows.json").read_text()),
        "audit": json.loads(
            (
                ROOT / ("data/processed/xiiotid_stage1_v3b_audit_band/audit_attack_windows.json")
            ).read_text()
        ),
    }
    context = {}
    flags = {}
    for band in windows:
        lookup = {(w["session"], w["window_start_ns"]): w for w in windows[band]}
        normal, attack = samples[f"{band}_normal"], samples[f"{band}_attack"]
        info = [lookup[(s.metadata["session_key"], s.metadata["window_start_ns"])] for s in attack]
        flags[band] = [
            np.asarray(
                [key in set(w["attack_endpoint_keys"]) for key in s.metadata["endpoint_keys"]]
            )
            for s, w in zip(attack, info, strict=True)
        ]
        context[band] = {
            "normal_nodes": np.asarray([s.node_features.shape[0] for s in normal]),
            "attack_nodes": np.asarray([s.node_features.shape[0] for s in attack]),
            "normal_dates": np.asarray([s.metadata["session_key"] for s in normal]),
            "attack_dates": np.asarray([s.metadata["session_key"] for s in attack]),
            "attack_fraction": np.asarray([w["attack_fraction"] for w in info]),
        }
    node_scores: dict[str, dict[str, list[torch.Tensor]]] = {
        "mahalanobis": {
            role: [detector.scorer.detector.score(t) for t in rows]
            for role, rows in states.items()
            if role != "train"
        },
    }
    reference = torch.cat(states["train"])
    for whiten in (False, True):
        local = LocalSupport.fit(reference, k=20, whiten=whiten)
        names = {
            kind: f"{kind}20_{'whitened' if whiten else 'euclidean'}" for kind in ("knn", "lof")
        }
        for name in names.values():
            node_scores[name] = {}
        for role, rows in states.items():
            if role == "train":
                continue
            scored = local.score(torch.cat(rows))
            lengths = [len(t) for t in rows]
            for kind, name in names.items():
                node_scores[name][role] = list(scored[kind].split(lengths))
        print(json.dumps({"seed": seed, "phase": "local_scored", "whiten": whiten}), flush=True)

    # Availability rarity: a label-free diagnostic of whether scoring is driven
    # by missing-feature patterns rather than behavioural values.
    train_patterns: dict[tuple[bool, ...], int] = {}
    for sample in samples["train"]:
        for row in sample.node_available.tolist():
            key = tuple(row)
            train_patterns[key] = train_patterns.get(key, 0) + 1
    node_scores["availability_rarity"] = {}
    for role, graphs in samples.items():
        if role != "train":
            node_scores["availability_rarity"][role] = [
                torch.tensor(
                    [
                        -np.log(
                            (train_patterns.get(tuple(row), 0) + 1)
                            / (len(reference) + len(train_patterns))
                        )
                        for row in s.node_available.tolist()
                    ],
                    dtype=torch.float64,
                )
                for s in graphs
            ]

    methods, scored_arrays = {}, {}
    for name, roles in node_scores.items():
        for how in ("top3_mean", "max"):
            key = f"{name}|{how}"
            graph = {role: aggregate(values, how) for role, values in roles.items()}
            scored_arrays[key] = {role: torch.from_numpy(scores) for role, scores in graph.items()}
            methods[key] = {}
            for band in ("test", "audit"):
                normal, attack = graph[f"{band}_normal"], graph[f"{band}_attack"]
                summary = detection_summary(graph["calibration"], normal, attack, FPR_TARGETS)
                summary.update(stratified_aurocs(normal, attack, **context[band]))
                threshold = summary["at_calibrated_fpr"]["0.05"]["threshold"]
                missed = attack <= threshold
                localized = np.asarray(
                    [
                        bool(f[int(torch.argmax(v))])
                        for v, f in zip(roles[f"{band}_attack"], flags[band], strict=True)
                    ]
                )
                attack_max = np.asarray(
                    [
                        float(v[torch.from_numpy(f)].max())
                        for v, f in zip(roles[f"{band}_attack"], flags[band], strict=True)
                    ]
                )
                summary["miss_analysis"] = {
                    "missed": int(missed.sum()),
                    "missed_with_attack_top1_fraction": float(localized[missed].mean()),
                    "all_attack_top1_fraction": float(localized.mean()),
                    "attack_node_max_above_window_threshold_but_missed": int(
                        ((attack_max > threshold) & missed).sum()
                    ),
                    "normal_score_quantiles": quantiles(normal),
                    "attack_score_quantiles": quantiles(attack),
                    "missed_attack_record_fraction_quantiles": quantiles(
                        context[band]["attack_fraction"][missed]
                    ),
                }
                categories = sorted(
                    {
                        c
                        for sample in samples[f"{band}_attack"]
                        for c in sample.metadata["attack_categories"]
                    }
                )
                summary["per_category"] = {}
                for category in categories:
                    mask = np.asarray(
                        [
                            category in s.metadata["attack_categories"]
                            for s in samples[f"{band}_attack"]
                        ]
                    )
                    summary["per_category"][category] = {
                        "windows": int(mask.sum()),
                        "tpr@0.05": float((~missed[mask]).mean()),
                    }
                methods[key][band] = summary
            print(
                json.dumps(
                    {
                        "method": key,
                        **{
                            band: {
                                "auroc": round(methods[key][band]["auroc"], 4),
                                **{
                                    m: round(methods[key][band]["at_calibrated_fpr"]["0.05"][m], 4)
                                    for m in ("tpr", "fpr")
                                },
                            }
                            for band in ("test", "audit")
                        },
                    }
                ),
                flush=True,
            )
    report = {
        "status": "exploratory_mechanism_study_consumed_evaluation_data",
        "seed": seed,
        "fixed_design": {
            "local_k": 20,
            "normal_reference": "train_normal",
            "threshold_source": "calibration_normal",
            "no_encoder_training": True,
        },
        "fingerprints": fingerprints,
        "methods": methods,
    }
    write_json(report_path, report)
    with (output / f"mechanism_scores_seed_{seed}.pt").open("xb") as stream:
        torch.save({"graph": scored_arrays, "node": node_scores}, stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    run(args.seed, args.output_dir)


if __name__ == "__main__":
    main()
