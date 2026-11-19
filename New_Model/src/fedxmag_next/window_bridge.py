"""Bridge frozen connection energies back to actual whole-minute alarms.

Full labelled connection tables determine window labels, including attacks
missed by row sampling. The frozen source model scores every prepared target
row; a window's energy is its top-three connection mean. Calibration receives
only actual prior wholly normal windows, not filtered background fragments.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .daily_detector import DailyDeepEnergyDetector
from .detector import conformal_threshold
from .experiments import sha256, write_json
from .record_data import XIIOTID_COLUMNS
from .record_prequential import score_methods


def aggregate_connections(
    values: np.ndarray,
    reference: np.ndarray,
    method: str,
    *,
    reference_sorted: bool = False,
) -> float:
    if method == "top3":
        return float(np.sort(values)[-3:].mean())
    # Independent normal-selection rows define conservative upper-tail ranks;
    # window calibration remains separate, with no independence assumption.
    reference = reference if reference_sorted else np.sort(reference)
    pvalue = (1 + len(reference) - np.searchsorted(reference, values, side="left")) / (
        len(reference) + 1
    )
    if method == "tail_scan":
        # Empirical ranks saturate above the largest reference energy. Use an
        # exponential mean-excess extension fitted to normal selection only.
        # These are estimated tail scores, not finite-sample conformal p-values;
        # separate whole-normal-window calibration still makes final decisions.
        threshold = float(np.quantile(reference, 0.95))
        excess = reference[reference > threshold] - threshold
        scale = max(
            float(excess.mean()) if len(excess) else 0.0, float(np.std(reference)) * 1e-3, 1e-6
        )
        probability = (1 + np.sum(reference >= threshold)) / (len(reference) + 1)
        surprise = -np.log(pvalue)
        beyond = values > threshold
        surprise[beyond] = -np.log(probability) + (values[beyond] - threshold) / scale
        rank = np.arange(1, len(values) + 1) / len(values)
        return float(np.max(np.log(rank) + np.sort(surprise)[::-1]))
    if method == "rank_mean":
        return float(-np.log(pvalue).mean())
    if method == "rank_scan":
        rank = np.arange(1, len(values) + 1) / len(values)
        return float(np.max(np.log(rank) - np.log(np.sort(pvalue))))
    raise ValueError(f"Unknown connection aggregation: {method}")


def replay_windows(
    energy: np.ndarray,
    observed: np.ndarray,
    labels: np.ndarray,
    captures: np.ndarray,
    source_scores: list,
    *,
    capacity: int = 20,
    policy: str = "capture_reset",
    fpr: float = 0.05,
    feedback_energy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feedback = energy if feedback_energy is None else np.asarray(feedback_energy)
    if feedback.shape != energy.shape or not np.isfinite(feedback).all():
        raise ValueError("Window feedback energies must be finite and aligned")
    history = {}
    continuous = deque(source_scores[-capacity:], maxlen=capacity)
    flags, valid = np.zeros(len(energy), dtype=bool), np.zeros(len(energy), dtype=bool)
    for i, capture in enumerate(captures):
        if capture not in history:
            history[capture] = (
                continuous
                if policy == "continuous"
                else deque(source_scores[-capacity:], maxlen=capacity)
            )
        threshold = conformal_threshold(history[capture], fpr)
        valid[i] = bool(observed[i] and threshold is not None)
        flags[i] = bool(energy[i] > threshold) if valid[i] else False
        if observed[i] and labels[i] == 0:
            history[capture].append(feedback[i])
    return flags, valid


def census_sources(root: Path, dataset: str) -> list[Path]:
    if dataset == "cic":
        return [
            root / f"data/processed/cic_ids2017_conn/labelled/{day}.csv"
            for day in ("Monday", "Thursday", "Friday")
        ]
    if dataset == "xiiotid":
        return [root / "data/X-IIoTID/X-IIoTID dataset.csv"]
    if dataset == "ton":
        return sorted(
            (root / "data/TON-IoT/Processed_datasets/Processed_Network_dataset").glob(
                "Network_dataset_*.csv"
            )
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def complete_role_windows(timestamps: np.ndarray, rows: np.ndarray, truth: dict) -> list[int]:
    all_minutes, all_counts = np.unique(timestamps // 60, return_counts=True)
    role_minutes, role_counts = np.unique(timestamps[rows] // 60, return_counts=True)
    positions = np.searchsorted(all_minutes, role_minutes)
    return [
        int(minute)
        for minute, count, total in zip(
            role_minutes, role_counts, all_counts[positions], strict=True
        )
        if count == total and int(minute) in truth and truth[int(minute)]["label"] == 0
    ]


def census(paths: list[Path]) -> dict:
    windows = {}
    for path in paths:
        for frame in pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            chunksize=250000,
            usecols=lambda name: (
                XIIOTID_COLUMNS.get(name, name) in {"ts", "duration", "label", "type"}
            ),
        ):
            frame = frame.rename(columns=XIIOTID_COLUMNS)
            duration = pd.to_numeric(frame["duration"], errors="coerce").fillna(0).clip(lower=0)
            start_time = pd.to_numeric(frame["ts"], errors="coerce").to_numpy()
            valid = np.isfinite(start_time)
            if not valid.any():
                continue
            minute = np.ceil(start_time[valid] + duration.to_numpy()[valid]).astype(np.int64) // 60
            labels = (
                frame["label"]
                .str.strip()
                .str.casefold()
                .replace({"normal": "0", "benign": "0", "false": "0", "attack": "1", "true": "1"})
            )
            labels = pd.to_numeric(labels, errors="raise").to_numpy()[valid]
            if not np.isin(labels, (0, 1)).all():
                raise ValueError("Invalid whole-window ground-truth labels")
            types = frame["type"].to_numpy()[valid]
            order = np.argsort(minute, kind="stable")
            starts = np.flatnonzero(np.r_[True, np.diff(minute[order]) != 0])
            for start, stop in zip(starts, np.r_[starts[1:], len(order)], strict=True):
                members = order[start:stop]
                key = int(minute[members[0]])
                state = windows.setdefault(key, {"label": 0, "records": 0, "types": set()})
                state["label"] = max(state["label"], int(labels[members].max()))
                state["records"] += len(members)
                state["types"].update(types[members][labels[members] == 1])
    for state in windows.values():
        state["types"] = sorted(state["types"])
    return windows


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_path, artifact = Path(args.data).resolve(), Path(args.detector).resolve()
    with np.load(data_path) as payload:
        data = {key: payload[key] for key in payload.files}
    dataset = str(data.get("dataset", "ton"))
    offset = int(data.get("utc_offset_seconds", 36000))
    tables = census_sources(root, dataset)
    if not tables:
        raise ValueError("No full connection tables for the window census")
    truth = census(tables)
    ts, labels = data["observed_at"], data["label"]
    if dataset == "xiiotid":
        days = np.unique((data["timestamp"] + offset) // 86400)
        cutoff, end = (
            int(days[int(0.6 * len(days))]) * 86400 - offset,
            (int(days[-1]) + 1) * 86400 - offset,
        )
    else:
        dates = ("2019-04-23", "2019-05-01") if dataset == "ton" else ("2017-07-06", "2017-07-08")
        cutoff, end = [int(np.datetime64(day, "s").astype(np.int64)) - offset for day in dates]
    normal = np.flatnonzero((ts < cutoff) & (labels == 0))
    normal = normal[np.argsort(ts[normal], kind="stable")]
    selection = normal[int(0.7 * len(normal)) : int(0.85 * len(normal))]
    calibration = normal[int(0.85 * len(normal)) :]
    target = np.flatnonzero((ts >= cutoff) & (ts < end))
    detector = DailyDeepEnergyDetector.load(artifact)
    explicit_roles = detector.state.get("source_role_indices")
    if explicit_roles is not None:
        if detector.state["source_data_sha256"] != sha256(data_path):
            raise ValueError("Explicit source roles refer to different observations")
        selection, calibration, fitted = [
            explicit_roles[name].numpy() for name in ("selection", "calibration", "train")
        ]
        if any(
            np.intersect1d(left, right).size
            for left, right in (
                (fitted, selection),
                (fitted, calibration),
                (selection, calibration),
            )
        ):
            raise ValueError("Source roles overlap")
        if any(ts[rows].max() >= ts[target].min() for rows in (selection, calibration, fitted)):
            raise ValueError("Source role contains future target information")
    elif detector.state["latest_reference_observed_at"] >= int(ts[calibration].min()):
        raise ValueError("Frozen source detector includes its calibration observations")
    if detector.feature_names != tuple(data["feature_names"]):
        raise ValueError("Source detector and window data schemas differ")
    features = torch.from_numpy(data["features"])
    if explicit_roles is None:
        selection = selection[ts[selection] > detector.state["latest_reference_observed_at"]]
    if not len(selection):
        raise ValueError("No independent normal selection for rank reference")
    detector.upgrade_knn_precision(features[selection])
    write_json(
        output / "registration.json",
        {
            "status": "actual_whole_window_development",
            "args": vars(args),
            "data_sha256": sha256(data_path),
            "detector_sha256": sha256(artifact),
            "census_sha256": {path.name: sha256(path) for path in tables},
            "dataset": dataset,
            "query_labels": "any attack in the complete labelled table, not just sampled rows",
            "model": "frozen source-only; raw controls use same observed rows",
            "aggregation": args.aggregation,
            "rank_reference": "independent source normal selection; source roles precede target",
            "calibration": "source held-out windows, then strictly prior actual normal windows",
            "fpr_target": 0.05,
            "normal_window_buffer": args.buffer,
            "history_policy": args.history_policy,
        },
    )
    combined = np.concatenate((calibration, target))
    print(json.dumps({"phase": "window_source_scoring", "records": len(combined)}), flush=True)
    row_scores = score_methods(detector, features[combined])
    row_reference = score_methods(detector, features[selection])
    minutes = ts[combined] // 60
    grouped = defaultdict(list)
    for position, minute in enumerate(minutes):
        grouped[int(minute)].append(position)
    # A role can precede training, so both boundaries can cut a minute. Retain
    # only windows whose complete observed row set belongs to calibration.
    initial = complete_role_windows(ts, calibration, truth)
    queries = [minute for minute in sorted(truth) if cutoff // 60 <= minute < end // 60]
    all_windows = initial + queries
    windows_scores = {name: {} for name in row_scores}
    for name, values in row_scores.items():
        for minute, members in grouped.items():
            windows_scores[name][minute] = aggregate_connections(
                values[members], row_reference[name], args.aggregation
            )
    windows_scores["record_count"] = {
        minute: float(np.log1p(truth[minute]["records"])) for minute in all_windows
    }
    windows_scores["sampled_record_count"] = {
        minute: float(np.log1p(len(grouped.get(minute, [])))) for minute in all_windows
    }
    results, archive = {}, {}
    for name, values in windows_scores.items():
        source_scores = [
            values[minute] for minute in initial if minute in values and truth[minute]["label"] == 0
        ]
        source_minutes = [
            minute for minute in initial if minute in values and truth[minute]["label"] == 0
        ]
        target_labels = np.asarray([truth[minute]["label"] for minute in queries])
        energy = np.asarray([values.get(minute, 0.0) for minute in queries])
        observed = np.asarray([minute in values for minute in queries])
        captures = np.asarray([(minute * 60 + offset) // 86400 for minute in queries])
        flags, supported = replay_windows(
            energy,
            observed,
            target_labels,
            captures,
            source_scores,
            capacity=args.buffer,
            policy=args.history_policy,
        )
        normal_count = int(np.sum(supported & (target_labels == 0)))
        attack_count = int(np.sum(supported & (target_labels == 1)))
        result = {
            "coverage": float(supported.mean()),
            "total_windows": len(queries),
            "evaluated_windows": int(supported.sum()),
            "total_normal_windows": int(np.sum(target_labels == 0)),
            "total_attack_windows": int(np.sum(target_labels == 1)),
            "end_to_end_attack_recall": float(flags[target_labels == 1].mean())
            if np.any(target_labels == 1)
            else None,
            "normal_windows": normal_count,
            "attack_windows": attack_count,
            "auroc": float(roc_auc_score(target_labels[supported], energy[supported]))
            if normal_count and attack_count
            else None,
            "tpr": float(flags[supported & (target_labels == 1)].mean()) if attack_count else None,
            "fpr": float(flags[supported & (target_labels == 0)].mean()) if normal_count else None,
        }
        types = sorted({kind for minute in queries for kind in truth[minute]["types"]})
        result["per_type_tpr"] = {
            kind: {
                "available_windows": int(sum(kind in truth[minute]["types"] for minute in queries)),
                "evaluated_windows": int(
                    sum(
                        supported[i] and kind in truth[minute]["types"]
                        for i, minute in enumerate(queries)
                    )
                ),
                "tpr": float(
                    flags[
                        np.asarray(
                            [
                                supported[i] and kind in truth[minute]["types"]
                                for i, minute in enumerate(queries)
                            ]
                        )
                    ].mean()
                )
                if any(
                    supported[i] and kind in truth[minute]["types"]
                    for i, minute in enumerate(queries)
                )
                else None,
            }
            for kind in types
        }
        results[name] = result
        archive[name] = {
            "energy": torch.from_numpy(energy),
            "flags": torch.from_numpy(flags),
            "valid": torch.from_numpy(supported),
            "source_calibration": torch.tensor(source_scores, dtype=torch.float64),
            "source_calibration_minutes": torch.tensor(source_minutes, dtype=torch.int64),
            "observed": torch.from_numpy(observed),
        }
        print(
            json.dumps(
                {
                    "method": name,
                    "metrics": {key: result[key] for key in ("auroc", "tpr", "fpr", "coverage")},
                }
            ),
            flush=True,
        )
    write_json(
        output / "report.json",
        {
            "status": "actual_whole_window_development",
            "dataset": dataset,
            "history_policy": args.history_policy,
            "aggregation": args.aggregation,
            "methods": results,
            "source_calibration_windows": len(initial),
        },
    )
    torch.save(
        {
            "methods": archive,
            "minutes": torch.tensor(queries),
            "labels": torch.tensor([truth[minute]["label"] for minute in queries]),
            "types": [truth[minute]["types"] for minute in queries],
        },
        output / "scores.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--buffer", type=int, default=20)
    parser.add_argument(
        "--history-policy", choices=("capture_reset", "continuous"), default="capture_reset"
    )
    parser.add_argument(
        "--aggregation", choices=("top3", "rank_scan", "rank_mean", "tail_scan"), default="top3"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
