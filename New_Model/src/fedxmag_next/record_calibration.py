"""Causal replay of externally verified normal-connection calibration.

All records completing at the same second are scored before any feedback from
that second is admitted. Alerts never decide which observations are normal.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from .detector import conformal_threshold
from .experiments import sha256, write_json
from .record_experiment import evaluate


def calibration_indices(data: dict) -> np.ndarray:
    timestamp, labels, capture = data["observed_at"], data["label"], data["capture"]
    dataset = str(data.get("dataset", "ton"))
    offset = int(data.get("utc_offset_seconds", 10 * 3600))
    start_day, end_day = (
        ("2019-04-23", "2019-05-01") if dataset == "ton" else ("2017-07-06", "2017-07-08")
    )
    cutoff = int(np.datetime64(start_day, "s").astype(np.int64)) - offset
    end = int(np.datetime64(end_day, "s").astype(np.int64)) - offset
    target = np.flatnonzero((timestamp >= cutoff) & (timestamp < end))
    selected = []
    for day in sorted(set(capture[target])):
        members = target[capture[target] == day]
        ordered = members[np.argsort(timestamp[members], kind="stable")]
        first = np.quantile(timestamp[ordered], 0.2, method="lower")
        second = np.quantile(timestamp[ordered], 0.4, method="lower")
        selected.extend(
            ordered[
                (timestamp[ordered] > first)
                & (timestamp[ordered] <= second)
                & (labels[ordered] == 0)
            ].tolist()
        )
    return np.asarray(selected, dtype=int)


def replay(
    values: np.ndarray,
    labels: np.ndarray,
    captures: np.ndarray,
    timestamps: np.ndarray,
    cal_values: np.ndarray,
    cal_capture: np.ndarray,
    cal_timestamp: np.ndarray,
    capacity: int,
    *,
    fpr: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    history = defaultdict(lambda: deque(maxlen=capacity))
    threshold = {}
    for capture in set(captures):
        members = np.flatnonzero(cal_capture == capture)
        ordered = members[np.argsort(cal_timestamp[members], kind="stable")]
        if len(ordered) and np.max(cal_timestamp[ordered]) >= np.min(
            timestamps[captures == capture]
        ):
            raise ValueError("Calibration feedback must strictly precede evaluation")
        history[capture].extend(cal_values[ordered].tolist())
        threshold[capture] = conformal_threshold(history[capture], fpr)
    flags = np.zeros(len(values), dtype=bool)
    valid = np.zeros(len(values), dtype=bool)
    order = np.lexsort((timestamps, captures))
    keys = [(int(captures[index]), int(timestamps[index])) for index in order]
    start = 0
    while start < len(order):
        capture, timestamp = keys[start]
        stop = start + 1
        while stop < len(order) and keys[stop] == (capture, timestamp):
            stop += 1
        members = order[start:stop]
        current = threshold[capture]
        if current is not None:
            flags[members] = values[members] > current
            valid[members] = True
        admitted = members[labels[members] == 0]
        if len(admitted):
            history[capture].extend(values[admitted].tolist())
            threshold[capture] = conformal_threshold(history[capture], fpr)
        start = stop
    return flags, valid


def cluster_difference(left: tuple, right: tuple, labels: np.ndarray, captures: np.ndarray) -> dict:
    lflags, lvalid = left
    rflags, rvalid = right
    valid = lvalid & rvalid
    groups = sorted(set(captures[valid]))
    generator = np.random.default_rng(20261105)
    differences = {"tpr": [], "fpr": []}
    totals = {}
    for name, label in (("tpr", 1), ("fpr", 0)):
        counts, increments = [], []
        for group in groups:
            chosen = valid & (labels == label) & (captures == group)
            counts.append(int(chosen.sum()))
            increments.append(float((lflags[chosen].astype(float) - rflags[chosen]).sum()))
        totals[name] = (np.asarray(counts), np.asarray(increments))
    for _ in range(2000):
        draw = generator.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
        for name, (counts, increments) in totals.items():
            denominator = counts @ draw
            if denominator:
                differences[name].append(float((increments @ draw) / denominator))
    return {
        name: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "captures": len(groups),
        }
        for name, values in differences.items()
    }


def run(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_path, score_path = Path(args.data).resolve(), Path(args.scores).resolve()
    with np.load(data_path) as payload:
        data = {key: payload[key] for key in payload.files}
    saved = torch.load(score_path, weights_only=True)
    selected = saved["evaluation_indices"].numpy()
    calibration = calibration_indices(data)
    labels, captures, timestamps, categories = [
        data[key][selected] for key in ("label", "capture", "observed_at", "type")
    ]
    write_json(
        output / "registration.json",
        {
            "status": "connection_level_causal_calibration_development",
            "args": vars(args),
            "data_sha256": sha256(data_path),
            "scores_sha256": sha256(score_path),
            "feedback": (
                "actual normal connection label after scoring; "
                "same-second feedback admitted only after whole group"
            ),
            "buffers": [20, 100, 200],
            "weights_and_energy_scores_unchanged": True,
            "uncertainty": (
                "paired capture bootstrap of fixed replay predictions; no model-refit uncertainty"
            ),
        },
    )
    report = {"status": "connection_level_causal_calibration_development", "policies": {}}
    traces = {}
    for capacity in (20, 100, 200):
        results = {}
        for method, arrays in saved["scores"]["commissioned"].items():
            cal_values, values = arrays["calibration"].numpy(), arrays["evaluation"].numpy()
            if len(cal_values) != len(calibration):
                raise ValueError("Calibration role reconstruction differs from stored scores")
            flags, valid = replay(
                values,
                labels,
                captures,
                timestamps,
                cal_values,
                data["capture"][calibration],
                data["observed_at"][calibration],
                capacity,
            )
            # evaluate() computes ranking using original energies; operational
            # decisions are replaced by the causal replay flags below.
            result = evaluate(
                values[valid],
                labels[valid],
                categories[valid],
                captures[valid],
                {int(c): 0.0 for c in set(captures[valid])},
            )
            result["coverage"] = float(valid.mean())
            result["tpr"] = float(flags[valid & (labels == 1)].mean())
            result["fpr"] = float(flags[valid & (labels == 0)].mean())
            result["normal"] = int(np.sum(valid & (labels == 0)))
            result["attack"] = int(np.sum(valid & (labels == 1)))
            result["records"] = int(valid.sum())
            for capture in sorted(set(captures)):
                chosen = valid & (captures == capture)
                result["per_capture"][str(int(capture))] = {
                    "normal": int(np.sum(chosen & (labels == 0))),
                    "attack": int(np.sum(chosen & (labels == 1))),
                    "fpr": float(flags[chosen & (labels == 0)].mean())
                    if np.any(chosen & (labels == 0))
                    else None,
                    "tpr": float(flags[chosen & (labels == 1)].mean())
                    if np.any(chosen & (labels == 1))
                    else None,
                }
            result["per_type_tpr"] = {
                name: {
                    "records": int(np.sum(valid & (labels == 1) & (categories == name))),
                    "tpr": float(flags[valid & (labels == 1) & (categories == name)].mean()),
                }
                for name in sorted(set(categories[valid & (labels == 1)]))
            }
            results[method] = result
            traces[(capacity, method)] = (flags, valid)
        report["policies"][str(capacity)] = results
        print(
            json.dumps(
                {
                    "buffer": capacity,
                    "methods": {
                        name: {key: value[key] for key in ("auroc", "tpr", "fpr", "coverage")}
                        for name, value in results.items()
                    },
                }
            ),
            flush=True,
        )
    report["capture_bootstrap_200"] = {
        f"{name}_minus_{baseline}": cluster_difference(
            traces[(200, name)], traces[(200, baseline)], labels, captures
        )
        for name in ("deep_mahalanobis_m8_ensemble", "raw_plus_deep_m8")
        for baseline in ("raw_mahalanobis_m8", "raw_knn")
    }
    write_json(output / "report.json", report)
    torch.save(
        {
            f"{capacity}_{name}": {
                "flags": torch.from_numpy(flags),
                "valid": torch.from_numpy(valid),
            }
            for (capacity, name), (flags, valid) in traces.items()
        },
        output / "traces.pt",
    )
    print(json.dumps(report["capture_bootstrap_200"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
