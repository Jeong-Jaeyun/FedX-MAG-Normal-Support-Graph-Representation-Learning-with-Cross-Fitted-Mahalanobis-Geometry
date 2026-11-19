"""Matched endpoint-conditioned normal geometry, unchanged window labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .entity_support import EntitySupportDetector, fit_entity_state
from .experiments import sha256, write_json
from .window_bridge import aggregate_connections, replay_windows
from .window_replay import run as evaluate_windows


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.workers)
    data_path, key_path, artifact, control, output = map(
        lambda p: Path(p).resolve(),
        (args.data, args.keys, args.detector, args.control, args.output),
    )
    base = torch.load(artifact, map_location="cpu", weights_only=True)
    origin = json.loads((control / "registration.json").read_text())
    data_hash = sha256(data_path)
    if data_hash != base["source_data_sha256"] or data_hash != origin["data_sha256"]:
        raise ValueError("Entity support comparison changed observations")
    if sha256(artifact) != origin["detector_sha256"]:
        raise ValueError("Entity support comparison changed frozen encoder")
    with np.load(data_path) as payload:
        data = {k: payload[k] for k in payload.files}
    with np.load(key_path) as payload:
        keys = {k: payload[k] for k in payload.files}
    if not np.array_equal(data["signature"], keys["signature"]):
        raise ValueError("Endpoint keys are not aligned to queries")
    if tuple(data["feature_names"]) != tuple(base["feature_names"]):
        raise ValueError("Entity support feature schema differs")
    stored = torch.load(control / "scores.pt", weights_only=True)
    roles = {k: v.numpy() for k, v in base["source_role_indices"].items()}
    target_minutes = stored["minutes"].numpy()
    if any(np.any(data["label"][ids] != 0) for ids in roles.values()):
        raise ValueError("Entity support training received attacks")
    if len(np.unique(np.concatenate(list(roles.values())))) != sum(map(len, roles.values())):
        raise ValueError("Entity support source roles overlap")
    if any(data["observed_at"][ids].max() >= target_minutes.min() * 60 for ids in roles.values()):
        raise ValueError("Entity support fit entered target interval")
    reference = roles["train"]
    reference = reference[np.argsort(data["signature"][reference])[:20000]]
    selected = roles["selection"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": str(data["dataset"]),
            "data_sha256": data_hash,
            "keys_sha256": sha256(key_path),
            "base_encoder_sha256": sha256(artifact),
            "control_scores_sha256": sha256(control / "scores.pt"),
            "census_sha256": origin["census_sha256"],
            "fit": (
                "frozen dependency coefficients; same capped source-train rows; "
                "normal selection units"
            ),
            "identity": (
                "same-network entity equality routes geometry; hashes never neural coordinates"
            ),
            "local_support": ">=32 source train AND selection rows per endpoint; otherwise global",
            "factorial": "raw/deep x global/source/destination x 1/8 modes; upper energy only",
            "precision": (
                "original float32 input units, copied DNN coefficients evaluated in float64"
            ),
            "aggregation": origin["aggregation"],
            "feedback": (
                "same whole source calibration windows; "
                "prediction before verified-normal feedback"
            ),
            "comparison": (
                "same observations, census and roles within each run; all old controls preserved"
            ),
        },
    )
    features = torch.from_numpy(data["features"])
    def subset_keys(ids):
        return {k: keys[k][ids] for k in ("source", "destination")}

    state = fit_entity_state(
        base, features[reference], features[selected], subset_keys(reference), subset_keys(selected)
    )
    state["source_reference_indices"] = torch.from_numpy(reference.copy())
    torch.save(state, output / "entity_detector.pt")
    print(
        "local normal supports",
        {name: len(bank) - 1 for name, bank in state["banks"].items()},
        flush=True,
    )
    minutes = data["observed_at"] // 60
    source_minutes = stored["methods"]["raw_knn"]["source_calibration_minutes"].numpy()
    needed = np.unique(
        np.concatenate(
            (selected, roles["calibration"], np.flatnonzero(np.isin(minutes, target_minutes)))
        )
    )
    mapping = np.full(len(minutes), -1, dtype=np.int64)
    mapping[needed] = np.arange(len(needed))
    detector = EntitySupportDetector(state)
    print("entity support query rows", len(needed), flush=True)
    scores = detector.score(features[needed], subset_keys(needed))
    audit_rows = np.linspace(0, len(needed) - 1, 128, dtype=np.int64)
    restored = EntitySupportDetector.load(output / "entity_detector.pt")
    reproduced = restored.score(features[needed[audit_rows]], subset_keys(needed[audit_rows]))
    audit = {}
    for name, values in scores.items():
        expected = values[audit_rows]
        error = float(((reproduced[name] - expected).abs() / (1 + expected.abs())).max())
        if error > 1e-5:
            raise ValueError(f"Restored entity support differs: {name}/{error}")
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
        raise ValueError("A whole source calibration minute mixes roles")
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
        normal = values[mapping[selected]]

        def aggregate(t, values=values, normal=normal):
            positions = mapping[groups[int(t)]]
            if np.any(positions < 0):
                raise ValueError("A query minute was only partially scored")
            return aggregate_connections(values[positions], normal, origin["aggregation"])

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
                raise ValueError(f"Matched entity control differs: {name}/{key}")
    report = json.loads((output / "evaluation/report.json").read_text())
    write_json(
        output / "report.json",
        {
            **report,
            "all_control_decisions_exact_at_matched_buffer": True,
            "restore_scaled_errors": audit,
            "neural_weights_frozen": True,
            "local_support_counts": {name: len(bank) - 1 for name, bank in state["banks"].items()},
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
    for name in ("data", "keys", "detector", "control", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--buffer", type=int, default=39)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
