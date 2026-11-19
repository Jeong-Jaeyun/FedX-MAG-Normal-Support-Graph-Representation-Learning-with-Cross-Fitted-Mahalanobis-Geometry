"""Paired nested endpoint rank evidence with frozen original edge energies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .daily_detector import DailyDeepEnergyDetector
from .experiments import sha256, write_json
from .hierarchical_scan import SCOPES, HierarchicalScanDetector, edge_scores, group_statistics
from .window_bridge import aggregate_connections, replay_windows
from .window_replay import run as evaluate_windows


def minute_groups(minutes):
    order = np.argsort(minutes, kind="stable")
    unique, starts, counts = np.unique(minutes[order], return_index=True, return_counts=True)
    return {
        int(t): order[start : start + count]
        for t, start, count in zip(unique, starts, counts, strict=True)
    }


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.workers)
    data_path, key_path, artifact, control, output = map(
        lambda p: Path(p).resolve(),
        (args.data, args.keys, args.detector, args.control, args.output),
    )
    base = DailyDeepEnergyDetector.load(artifact)
    origin = json.loads((control / "registration.json").read_text())
    data_hash = sha256(data_path)
    if data_hash != base.state["source_data_sha256"] or data_hash != origin["data_sha256"]:
        raise ValueError("Nested rank comparison changed observations")
    if sha256(artifact) != origin["detector_sha256"] or origin["aggregation"] != "rank_scan":
        raise ValueError("Nested rank comparison needs the original rank-scan encoder")
    with np.load(data_path) as payload:
        data = {k: payload[k] for k in payload.files}
    with np.load(key_path) as payload:
        keys = {k: payload[k] for k in ("signature", "source", "destination")}
    if not np.array_equal(data["signature"], keys["signature"]):
        raise ValueError("Nested rank keys are misaligned")
    if tuple(data["feature_names"]) != base.feature_names:
        raise ValueError("Nested rank input schema differs")
    stored = torch.load(control / "scores.pt", weights_only=True)
    target_minutes = stored["minutes"].numpy()
    roles = {k: v.numpy() for k, v in base.state["source_role_indices"].items()}
    if any(np.any(data["label"][v] != 0) for v in roles.values()):
        raise ValueError("Nested rank source roles contain attack rows")
    if len(np.unique(np.concatenate(list(roles.values())))) != sum(map(len, roles.values())):
        raise ValueError("Nested rank source roles overlap")
    if any(data["observed_at"][v].max() >= target_minutes.min() * 60 for v in roles.values()):
        raise ValueError("Nested rank fitting entered target interval")
    minutes = data["observed_at"] // 60
    target = np.flatnonzero(np.isin(minutes, target_minutes))
    # Match the exact original window_bridge order and forward batches.
    combined = np.concatenate((roles["calibration"], target))
    selected = roles["selection"]
    features = torch.from_numpy(data["features"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": str(data["dataset"]),
            "data_sha256": data_hash,
            "detector_sha256": sha256(artifact),
            "keys_sha256": sha256(key_path),
            "control_scores_sha256": sha256(control / "scores.pt"),
            "census_sha256": origin["census_sha256"],
            "fit": (
                "frozen original edge energies; nested within-endpoint then across-group rank_scan"
            ),
            "group_reference": (
                "only complete selection normal minutes, endpoint identity discarded"
            ),
            "scopes": list(SCOPES),
            "tie_tolerance": {"relative": 1e-6, "absolute": 1e-9},
            "feedback": (
                "same complete source calibration minutes, prediction before normal verification"
            ),
            "identity": (
                "current-minute equality grouping only; no numeric IP feature or learned GNN"
            ),
        },
    )
    print("nested rank edge queries", len(combined), flush=True)
    row_scores = edge_scores(base, features[combined])
    row_reference = edge_scores(base, features[selected])
    groups, selected_groups, all_groups = (
        minute_groups(minutes[combined]),
        minute_groups(minutes[selected]),
        minute_groups(minutes),
    )
    complete_selection = [t for t, ids in selected_groups.items() if len(ids) == len(all_groups[t])]
    if len(complete_selection) < 2:
        raise ValueError("Too few complete source selection windows for group references")
    source_minutes = stored["methods"]["raw_knn"]["source_calibration_minutes"].numpy()
    membership = np.zeros(len(minutes), dtype=bool)
    membership[roles["calibration"]] = True
    if any(not membership[all_groups[int(t)]].all() for t in source_minutes):
        raise ValueError("Nested calibration minute mixes roles")
    row_reference_sorted = {name: np.sort(v) for name, v in row_reference.items()}
    group_reference = {}
    for scope in SCOPES:
        samples = {name: [] for name in row_scores}
        for t in complete_selection:
            ids = selected_groups[t]
            grouped = group_statistics(
                {k: v[ids] for k, v in row_reference.items()},
                {k: keys[k][selected[ids]] for k in ("source", "destination")},
                scope,
                row_reference_sorted,
            )
            for name, values in grouped.items():
                samples[name].append(values)
        group_reference[scope] = {
            k: torch.from_numpy(np.sort(np.concatenate(v))) for k, v in samples.items()
        }
    state = {
        "format_version": 1,
        "base_encoder": base.state,
        "row_reference": {k: torch.from_numpy(v) for k, v in row_reference_sorted.items()},
        "group_reference": group_reference,
        "selection_reference_minutes": torch.tensor(complete_selection),
    }
    torch.save(state, output / "hierarchical_detector.pt")
    fitted = HierarchicalScanDetector(state)
    window_values = {}
    for t, positions in groups.items():
        if t not in target_minutes and t not in source_minutes:
            continue
        window_values[t] = fitted.aggregate(
            {k: v[positions] for k, v in row_scores.items()},
            {k: keys[k][combined[positions]] for k in ("source", "destination")},
        )
    # Recompute flat controls from these actual rows, rather than merely copying
    # old decisions. Exact scoring order should preserve every saved energy.
    flat_errors = {}
    for name, values in row_scores.items():
        normal = row_reference[name]
        for field, ts in (("energy", target_minutes), ("source_calibration", source_minutes)):
            actual = np.array(
                [
                    aggregate_connections(values[groups[int(t)]], normal, "rank_scan")
                    if int(t) in groups
                    else 0
                    for t in ts
                ]
            )
            expected = stored["methods"][name][field].numpy()
            error = float(np.max(np.abs(actual - expected) / (1 + np.abs(expected))))
            if error > 1e-8:
                raise ValueError(f"Nested flat control differs: {name}/{field}/{error}")
            flat_errors[f"{name}/{field}"] = error
    restored = HierarchicalScanDetector.load(output / "hierarchical_detector.pt")
    audit_ids = np.linspace(0, len(combined) - 1, 128, dtype=np.int64)
    rescored = edge_scores(restored.base, features[combined[audit_ids]])
    row_errors = {}
    for name, values in row_scores.items():
        expected = values[audit_ids]
        error = float(np.max(np.abs(rescored[name] - expected) / (1 + np.abs(expected))))
        if error > 1e-5:
            raise ValueError(f"Nested restored row inference differs: {name}/{error}")
        row_errors[name] = error
    window_error = 0.0
    audit_minutes = [int(t) for t in target_minutes if int(t) in groups]
    for i in np.linspace(0, len(audit_minutes) - 1, 16, dtype=np.int64):
        t = audit_minutes[i]
        ids = combined[groups[t]]
        actual = restored.score_window(
            features[ids], {k: keys[k][ids] for k in ("source", "destination")}
        )
        for name, value in actual.items():
            error = abs(value - window_values[t][name])
            if error > 1e-6:
                raise ValueError(f"Nested restored actual window differs: {t}/{name}/{error}")
            window_error = max(window_error, error)
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
    for name in next(iter(window_values.values())):
        archive[name] = {
            "energy": torch.tensor(
                [
                    window_values[int(t)][name] if int(t) in window_values else 0.0
                    for t in target_minutes
                ],
                dtype=torch.float64,
            ),
            "observed": torch.tensor([int(t) in window_values for t in target_minutes]),
            "source_calibration": torch.tensor(
                [window_values[int(t)][name] for t in source_minutes], dtype=torch.float64
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
                raise ValueError(f"Nested matched control differs: {name}/{key}")
    report = json.loads((output / "evaluation/report.json").read_text())
    write_json(
        output / "report.json",
        {
            **report,
            "all_control_decisions_exact_at_matched_buffer": True,
            "flat_control_scaled_errors": flat_errors,
            "restore_row_scaled_errors": row_errors,
            "restore_actual_window_max_error": window_error,
            "complete_selection_windows": len(complete_selection),
            "group_reference_counts": {
                scope: {name: len(v) for name, v in ref.items()}
                for scope, ref in group_reference.items()
            },
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
