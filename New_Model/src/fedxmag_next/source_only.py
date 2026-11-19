"""Fit a source-only normal deep-energy artifact for whole-window evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .experiments import sha256, write_json
from .record_prequential import build_source


def source_roles(data: dict, normal: np.ndarray, order: str, gap: int = 300) -> dict:
    if order == "train_selection_calibration":
        a, b = int(0.7 * len(normal)), int(0.85 * len(normal))
        roles = {"train": normal[:a], "selection": normal[a:b], "calibration": normal[b:]}
        sequence = ("train", "selection", "calibration")
    elif order == "calibration_selection_train":
        a, b = int(0.15 * len(normal)), int(0.3 * len(normal))
        roles = {"calibration": normal[:a], "selection": normal[a:b], "train": normal[b:]}
        sequence = ("calibration", "selection", "train")
    else:
        raise ValueError("Unknown source role order")
    # Current flow must start after the earlier role and its trailing context.
    # This prevents adjacent reserved windows from sharing 300-second histories.
    for previous, following in zip(sequence, sequence[1:], strict=False):
        boundary = int(data["observed_at"][roles[previous]].max()) + gap
        roles[following] = roles[following][data["timestamp"][roles[following]] > boundary]
        if len(roles[following]) < 32:
            raise ValueError("Too few normal source rows after temporal role purge")
    return roles


def physical_role_template(
    data: dict, artifact_path: str, gap: int, allow_feature_change: bool = False
) -> tuple[dict, dict]:
    """Preserve physical source intervals when a label-independent sample expands."""
    path = Path(artifact_path).resolve()
    state = torch.load(path, map_location="cpu", weights_only=True)
    registration = json.loads((path.parent / "registration.json").read_text())
    old_path = Path(registration["args"]["data"]).resolve()
    if sha256(old_path) != state["source_data_sha256"]:
        raise ValueError("Source role template input was changed")
    if gap != registration["args"]["history_gap"]:
        raise ValueError("Template must retain the original temporal purge")
    with np.load(old_path) as payload:
        old = {
            key: payload[key]
            for key in (
                "signature",
                "timestamp",
                "observed_at",
                "label",
                "dataset",
                "feature_names",
            )
        }
    if str(old["dataset"]) != str(data["dataset"]):
        raise ValueError("Source template dataset differs")
    same_observations = all(
        np.array_equal(old[key], data[key])
        for key in ("signature", "timestamp", "observed_at", "label")
    )
    if tuple(old["feature_names"]) != tuple(data["feature_names"]):
        if not allow_feature_change:
            raise ValueError("Sampling ablation must retain the original feature schema")
        if not same_observations:
            raise ValueError("Feature ablation must keep every observation and label identical")
        if not set(old["feature_names"]) <= set(data["feature_names"]):
            raise ValueError("Feature extension must retain all old fields")
    original = {name: rows.numpy() for name, rows in state["source_role_indices"].items()}
    intervals = {
        name: (int(old["observed_at"][rows].min()), int(old["observed_at"][rows].max()))
        for name, rows in original.items()
    }
    sequence = sorted(intervals, key=lambda name: intervals[name][0])
    policy = {
        "requested": "physical_time_template",
        "template_sha256": sha256(path),
        "template_data_sha256": sha256(old_path),
        "physical_intervals": intervals,
        "history_gap": gap,
        "allow_feature_change": allow_feature_change,
        "all_original_role_observations_retained": True,
        "membership_policy": "exact_saved_indices" if same_observations else "interval_expansion",
    }
    if same_observations:
        # Quantile splits may cut a completed-second tie. An inclusive time mask
        # would then add excluded rows, changing the capped reference and scale.
        for index, name in enumerate(sequence):
            rows = original[name]
            if len(rows) < 32 or np.any(data["label"][rows] != 0):
                raise ValueError("Invalid normal source membership in exact template")
            if index and np.any(data["timestamp"][rows] <= intervals[sequence[index - 1]][1] + gap):
                raise ValueError("Exact template membership violates its temporal purge")
        if len(np.unique(np.concatenate(list(original.values())))) != sum(
            map(len, original.values())
        ):
            raise ValueError("Exact template roles overlap")
        return {name: rows.copy() for name, rows in original.items()}, policy
    roles = {}
    for index, name in enumerate(sequence):
        lower, upper = intervals[name]
        mask = (
            (data["label"] == 0) & (data["observed_at"] >= lower) & (data["observed_at"] <= upper)
        )
        if index:
            previous = sequence[index - 1]
            mask &= data["timestamp"] > intervals[previous][1] + gap
        rows = np.flatnonzero(mask)
        rows = rows[np.argsort(data["observed_at"][rows], kind="stable")]
        mapped = np.searchsorted(data["signature"], old["signature"][original[name]])
        if np.any(mapped >= len(data["signature"])):
            raise ValueError("A template observation disappeared")
        if not np.array_equal(data["signature"][mapped], old["signature"][original[name]]):
            raise ValueError("Template observation signatures changed")
        if not np.array_equal(data["observed_at"][mapped], old["observed_at"][original[name]]):
            raise ValueError("Template observation times changed")
        if not np.array_equal(data["timestamp"][mapped], old["timestamp"][original[name]]):
            raise ValueError("Template flow starts changed")
        if not np.isin(mapped, rows).all() or len(rows) < 32:
            raise ValueError("Physical role template lost old membership or has too few normals")
        roles[name] = rows
    return roles, policy


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    output, path = Path(args.output).resolve(), Path(args.data).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with np.load(path) as payload:
        data = {key: payload[key] for key in payload.files}
    dataset = str(data.get("dataset", "ton"))
    offset = int(data.get("utc_offset_seconds", 36000))
    if dataset == "xiiotid":
        days = np.unique((data["timestamp"] + offset) // 86400)
        cutoff = int(days[int(0.6 * len(days))]) * 86400 - offset
    else:
        day = "2019-04-23" if dataset == "ton" else "2017-07-06"
        cutoff = int(np.datetime64(day, "s").astype(np.int64)) - offset
    normal = np.flatnonzero((data["observed_at"] < cutoff) & (data["label"] == 0))
    normal = normal[np.argsort(data["observed_at"][normal], kind="stable")]
    policy = {"requested": args.role_order}
    if getattr(args, "role_template", None):
        roles, policy = physical_role_template(
            data,
            args.role_template,
            args.history_gap,
            getattr(args, "allow_template_feature_change", False),
        )
    elif args.role_order == "auto_normal_windows":
        from .window_bridge import census, census_sources, complete_role_windows

        tables = census_sources(Path(args.root).resolve(), dataset)
        whole_truth = census(tables)
        policy["census_sha256"] = {p.name: sha256(p) for p in tables}
        policy["whole_normal_calibration_windows"] = {}
        roles = None
        for candidate in ("train_selection_calibration", "calibration_selection_train"):
            proposed = source_roles(data, normal, candidate, args.history_gap)
            rows = proposed["calibration"]
            count = len(complete_role_windows(data["observed_at"], rows, whole_truth))
            policy["whole_normal_calibration_windows"][candidate] = count
            if count >= args.minimum_source_windows:
                roles, policy["selected"] = proposed, candidate
                break
        if roles is None:
            raise ValueError("No source role order supplies enough actual held-out normal windows")
    else:
        roles = source_roles(data, normal, args.role_order, args.history_gap)
        policy["selected"] = args.role_order
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "data_sha256": sha256(path),
            "dataset": dataset,
            "source_cutoff": cutoff,
            "source_train_normals": len(roles["train"]),
            "selection_normals": len(roles["selection"]),
            "calibration_normals": len(roles["calibration"]),
            "fit": {
                "dependency": "source normals; synthetic feature-group discrimination selection",
                "association": "source normals; protocol/service-matched history-swap selection",
                "neural_transformations": "source normals; learned normal-transformation selection",
                "geometry_margin": (
                    "source normals; disjoint anchor/probe covariance training; "
                    "normal/synthetic-history geometry selection"
                ),
            }[getattr(args, "objective", "dependency")],
            "role_policy": policy,
        },
    )
    artifact = build_source(data, roles["train"], roles["selection"], args)
    artifact["source_role_indices"] = {name: torch.from_numpy(rows) for name, rows in roles.items()}
    artifact["source_data_sha256"] = sha256(path)
    artifact["source_role_policy"] = policy
    if any(data["observed_at"][rows].max() >= cutoff for rows in roles.values()):
        raise ValueError("A reserved source role crossed the first target date")
    torch.save(artifact, output / "source_detector.pt")
    write_json(
        output / "report.json",
        {
            "status": "source_normal_fit_complete",
            "artifact_sha256": sha256(output / "source_detector.pt"),
            "ignored_source_constant_features": []
            if artifact.get("input_keep") is None
            else [
                name
                for name, keep in zip(
                    artifact["feature_names"], artifact["input_keep"].tolist(), strict=True
                )
                if not keep
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--objective",
        choices=("dependency", "neural_transformations", "association", "geometry_margin"),
        default="dependency",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 41])
    parser.add_argument("--max-reference", type=int, default=20000)
    parser.add_argument(
        "--role-order",
        choices=(
            "train_selection_calibration",
            "calibration_selection_train",
            "auto_normal_windows",
        ),
        default="train_selection_calibration",
    )
    parser.add_argument("--history-gap", type=int, default=300)
    parser.add_argument("--role-template", help="Source artifact fixing physical role intervals")
    parser.add_argument("--allow-template-feature-change", action="store_true")
    parser.add_argument("--minimum-source-windows", type=int, default=20)
    parser.add_argument(
        "--constant-feature-policy",
        choices=("preserve", "ignore"),
        default="preserve",
        help="Fixed source-only policy for columns with source-train standard deviation <=1e-6",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
