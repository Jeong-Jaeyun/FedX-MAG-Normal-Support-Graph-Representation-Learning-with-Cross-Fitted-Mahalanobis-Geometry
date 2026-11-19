"""Paired frozen deep-neighbor whole-window development evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .deep_neighbors import DeepNeighborDetector, fit_neighbor_state
from .experiments import sha256, write_json
from .window_bridge import aggregate_connections, replay_windows
from .window_replay import run as evaluate_windows


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.workers)
    path, artifact, control, output = map(
        lambda value: Path(value).resolve(), (args.data, args.detector, args.control, args.output)
    )
    base = torch.load(artifact, map_location="cpu", weights_only=True)
    origin = json.loads((control / "registration.json").read_text())
    data_hash = sha256(path)
    if data_hash != base["source_data_sha256"] or data_hash != origin["data_sha256"]:
        raise ValueError("Neighbor model and paired controls use different observations")
    if sha256(artifact) != origin["detector_sha256"]:
        raise ValueError("Neighbor comparison uses another pretrained encoder")
    with np.load(path) as payload:
        data = {key: payload[key] for key in payload.files}
    if tuple(data["feature_names"]) != tuple(base["feature_names"]):
        raise ValueError("Neighbor input schema differs")
    roles = {key: ids.numpy() for key, ids in base["source_role_indices"].items()}
    stored = torch.load(control / "scores.pt", weights_only=True)
    target_minutes = stored["minutes"].numpy()
    if any(np.any(data["label"][ids] != 0) for ids in roles.values()):
        raise ValueError("Neighbor source roles received attacks")
    if any(data["observed_at"][ids].max() >= target_minutes.min() * 60 for ids in roles.values()):
        raise ValueError("Neighbor fit entered the target interval")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": str(data["dataset"]),
            "data_sha256": data_hash,
            "base_encoder_sha256": sha256(artifact),
            "control_scores_sha256": sha256(control / "scores.pt"),
            "census_sha256": origin["census_sha256"],
            "fit": "frozen DNN weights; exact existing10000 normal reference points; k20",
            "metrics": "latent Euclidean / frozen source1-mode Mahalanobis; raw whitened control",
            "forward_precision": "float64 coefficients copied from original float32 weights",
            "selection": "original source normals only, median/q95 distance units",
            "aggregation": origin["aggregation"],
            "feedback": "same source windows, then prior verified whole-normal feedback39",
            "comparison": "old energies preserved; all controls replayed explicitly at buffer39",
        },
    )
    features = torch.from_numpy(data["features"])
    state = fit_neighbor_state(base, features[roles["selection"]])
    torch.save(state, output / "neighbor_detector.pt")
    minutes = data["observed_at"] // 60
    source_minutes = stored["methods"]["raw_knn"]["source_calibration_minutes"].numpy()
    needed = np.unique(
        np.concatenate(
            (
                roles["selection"],
                roles["calibration"],
                np.flatnonzero(np.isin(minutes, target_minutes)),
            )
        )
    )
    mapping = np.full(len(minutes), -1, dtype=np.int64)
    mapping[needed] = np.arange(len(needed))
    detector = DeepNeighborDetector(state)
    print("neighbor query rows", len(needed), flush=True)
    scores = detector.score(features[needed])
    selected = mapping[roles["selection"]]
    audit_rows = np.linspace(0, len(needed) - 1, 128, dtype=np.int64)
    restored = DeepNeighborDetector.load(output / "neighbor_detector.pt")
    replay = restored.score(features[needed[audit_rows]])
    audit = {}
    for name, values in scores.items():
        expected = values[audit_rows]
        error = float(((replay[name] - expected).abs() / (1 + expected.abs())).max())
        if error > 1e-5:
            raise ValueError(f"Restored neighbor score differs: {name}/{error}")
        audit[name] = error
    order = np.argsort(minutes, kind="stable")
    unique, starts, counts = np.unique(minutes[order], return_index=True, return_counts=True)
    groups = {
        int(t): order[start : start + count]
        for t, start, count in zip(unique, starts, counts, strict=True)
    }
    member = np.zeros(len(minutes), dtype=bool)
    member[roles["calibration"]] = True
    if any(not member[groups[int(t)]].all() for t in source_minutes):
        raise ValueError("Source calibration minute mixes reserved roles")
    archive, matched = {}, {}
    for name, item in stored["methods"].items():
        flags, valid = replay_windows(
            item["energy"].numpy(),
            item["observed"].numpy(),
            stored["labels"].numpy(),
            np.zeros(len(target_minutes), dtype=int),
            item["source_calibration"].tolist(),
            capacity=args.buffer,
            policy="continuous",
            fpr=0.05,
        )
        archive[name] = {**item, "flags": torch.from_numpy(flags), "valid": torch.from_numpy(valid)}
        matched[name] = {"flags": archive[name]["flags"], "valid": archive[name]["valid"]}
    for name, values in scores.items():
        values = values.numpy()
        reference = values[selected]

        def aggregate(minute, score=values, normal=reference):
            positions = mapping[groups[int(minute)]]
            if np.any(positions < 0):
                raise ValueError("A whole query minute was only partially scored")
            return aggregate_connections(score[positions], normal, origin["aggregation"])

        archive[name] = {
            "energy": torch.tensor(
                [aggregate(t) if int(t) in groups else 0.0 for t in target_minutes],
                dtype=torch.float64,
            ),
            "observed": torch.tensor([int(t) in groups for t in target_minutes]),
            "source_calibration": torch.tensor(
                [aggregate(t) for t in source_minutes], dtype=torch.float64
            ),
            "source_calibration_minutes": torch.from_numpy(source_minutes.copy()),
        }
    torch.save({**stored, "methods": archive}, output / "scores.pt")
    evaluate_windows(
        argparse.Namespace(
            run=str(output),
            output=str(output / "evaluation"),
            buffer=args.buffer,
            fpr=0.05,
            history_policy="continuous",
        )
    )
    decisions = torch.load(output / "evaluation/decisions.pt", weights_only=True)
    for name, item in matched.items():
        for key, expected in item.items():
            if not torch.equal(decisions[name][key], expected):
                raise ValueError(f"Matched-capacity control differs: {name}/{key}")
    report = json.loads((output / "evaluation/report.json").read_text())
    write_json(
        output / "report.json",
        {
            **report,
            "all_control_decisions_exact_at_matched_buffer": True,
            "restore_scaled_errors": audit,
            "neural_weights_frozen": True,
        },
    )
    evaluate_windows(
        argparse.Namespace(
            run=str(output),
            output=str(output / "buffer50"),
            buffer=50,
            fpr=0.05,
            history_policy="continuous",
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--buffer", type=int, default=39)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
