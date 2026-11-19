"""Audit exported initial candidates through the integrated label-free query API."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .candidate_stream import CandidateStream
from .experiments import sha256, write_json
from .hierarchical_experiment import minute_groups


def rejected(call):
    try:
        call()
    except ValueError:
        return
    raise ValueError("Candidate API accepted an invalid feedback/checkpoint order")


@torch.no_grad()
def audit(record, output):
    packet_path = Path(record["model"])
    if sha256(packet_path) != record["model_sha256"]:
        raise ValueError("Registered initial candidate packet changed")
    packet = torch.load(packet_path, weights_only=True)
    risk_root = Path(packet["prototype_model"]["protocol"]["args"]["run"])
    registration = json.loads((risk_root / "registration.json").read_text())
    data_path = Path(registration["args"]["data"])
    keys_path = Path(registration["args"]["keys"])
    if sha256(data_path) != packet["data_sha256"]:
        raise ValueError("Exported candidate input observations changed")
    if sha256(keys_path) != registration["keys_sha256"]:
        raise ValueError("Exported candidate endpoint metadata changed")
    with np.load(data_path) as payload:
        features = torch.from_numpy(payload["features"])
        signature, times = payload["signature"], payload["observed_at"]
        names = tuple(payload["feature_names"])
    with np.load(keys_path) as payload:
        if not np.array_equal(signature, payload["signature"]):
            raise ValueError("Candidate query endpoints are misaligned")
        keys = {k: payload[k] for k in ("source", "destination", "port")}
    if names != tuple(packet["prototype_model"]["base_encoder"]["feature_names"]):
        raise ValueError("Candidate query feature order changed")
    expected_root = Path(record["report"]).parent
    expected = torch.load(expected_root / "scores.pt", weights_only=True)
    decisions = torch.load(expected_root / "evaluation/decisions.pt", weights_only=True)
    method = record["method"]
    minutes, labels = expected["minutes"].numpy(), expected["labels"].numpy()
    groups = minute_groups(times // 60)
    stream = CandidateStream(packet)
    rejected(lambda: stream.feedback(verified_normal=True))
    values, alarms, valid, thresholds = [], [], [], []
    restored_exact = False
    for i, minute in enumerate(minutes):
        ids = groups.get(int(minute), np.empty(0, dtype=np.int64))
        local_keys = {k: v[ids] for k, v in keys.items()}
        shadow = None
        if i == len(minutes) // 2:
            torch.save(stream.state_dict(), output / "midpoint.pt")
            shadow = CandidateStream.from_state_dict(
                torch.load(output / "midpoint.pt", weights_only=True)
            )
        result = stream.query(int(minute), features[ids], local_keys, times[ids])
        if i == 0:
            rejected(partial(stream.query, int(minute), features[ids], local_keys, times[ids]))
            rejected(stream.state_dict)
        if shadow is not None:
            repeated = shadow.query(int(minute), features[ids], local_keys, times[ids])
            if repeated != result:
                raise ValueError("Combined stream disk checkpoint changed actual inference")
            shadow.feedback(verified_normal=bool(labels[i] == 0))
            restored_exact = True
        stream.feedback(verified_normal=bool(labels[i] == 0))
        if shadow is not None:
            if list(shadow.base_history) != list(stream.base_history) or list(
                shadow.final_history
            ) != list(stream.final_history):
                raise ValueError("Restored combined stream changed actual normal feedback")
            stream = shadow
        values.append(result["energy"])
        thresholds.append(result["threshold"])
        alarms.append(result["alarm"])
        valid.append(result["valid"])
        if i % 500 == 0:
            print("candidate replay", packet_path.stem, i, len(minutes), flush=True)
    values, alarms, valid = np.array(values), np.array(alarms), np.array(valid)
    old_values = expected["methods"][method]["energy"].numpy()
    old_flags = decisions[method]["flags"].numpy()
    old_valid = decisions[method]["valid"].numpy()
    errors = np.abs(values - old_values)
    report = {
        "status": "exact_flags_from_exported_initial_state"
        if np.array_equal(alarms, old_flags) and np.array_equal(valid, old_valid)
        else "changed_flags_from_per_window_inference_investigate_before_adoption",
        "method": method,
        "packet": str(packet_path),
        "packet_sha256": record["model_sha256"],
        "data_sha256": packet["data_sha256"],
        "windows": len(minutes),
        "all_target_rows_recomputed_without_saved_energy_or_distance_cache": True,
        "flags_different": int((alarms != old_flags).sum()),
        "validity_different": int((valid != old_valid).sum()),
        "max_absolute_window_energy_error": float(errors.max()),
        "max_scaled_window_energy_error": float((errors / (1 + np.abs(old_values))).max()),
        "window_energies_bit_exact": bool(np.array_equal(values, old_values)),
        "midpoint_disk_restore_actual_inference_and_feedback_exact": restored_exact,
        "query_rejects_labels_and_requires_prior_feedback": True,
        "checkpoint_rejects_uncommitted_feedback": True,
        "coverage": float(valid.mean()),
        "tpr": float(alarms[labels == 1].mean()),
        "fpr": float(alarms[labels == 0].mean()),
        "per_type_tpr": {
            kind: float(alarms[[kind in group for group in expected["types"]]].mean())
            for kind in sorted({kind for group in expected["types"] for kind in group})
        },
    }
    torch.save(
        {
            "minutes": expected["minutes"],
            "labels": expected["labels"],
            "types": expected["types"],
            "energy": torch.from_numpy(values),
            "threshold": torch.tensor(thresholds),
            "flags": torch.from_numpy(alarms),
            "valid": torch.from_numpy(valid),
        },
        output / "predictions.pt",
    )
    write_json(output / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "per_type_tpr"}), flush=True)
    return report


def run(args):
    torch.set_num_threads(args.workers)
    registry = json.loads(Path(args.registry).read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for record in registry["candidates"]:
        if Path(record["model"]).stem not in args.models:
            continue
        destination = output / Path(record["model"]).stem
        destination.mkdir()
        reports.append(audit(record, destination))
    if len(reports) != len(args.models):
        raise ValueError("Some requested exported models were not audited")
    write_json(output / "report.json", {"candidates": reports})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--models", nargs="+", required=True)
    run(parser.parse_args())
