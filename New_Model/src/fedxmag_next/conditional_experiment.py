"""Score current-only DNN states conditionally on strictly past flow behavior.

Reuse a frozen current-only encoder and its exact source roles. Joint Gaussian
normal supports model [history,current], but only -2 log p(current|history)
triggers a query. All source-fitted parameters precede every target prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .conditional_detector import ConditionalCurrentDetector
from .conditional_support import HistoryConditionalSupport
from .daily_detector import DailyDeepEnergyDetector
from .experiments import sha256, write_json
from .record_experiment import corruption_groups
from .temporal_context import CONTEXT_FEATURES
from .window_replay import run as evaluate_windows


@torch.no_grad()
def scores(support: HistoryConditionalSupport, current, history, marginal=False):
    return torch.cat(
        [
            support.marginal_score(y) if marginal else support.score(y, h)
            for y, h in zip(current.split(8192), history.split(8192), strict=True)
        ]
    )


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    output, path, detector_path, control_path = (
        Path(args.output).resolve(),
        Path(args.data).resolve(),
        Path(args.detector).resolve(),
        Path(args.control).resolve(),
    )
    with np.load(path) as payload:
        data = {key: payload[key] for key in payload.files}
    detector = DailyDeepEnergyDetector.load(detector_path)
    corruption_groups(detector.feature_names)
    if any(name.startswith("context_") for name in detector.feature_names):
        raise ValueError("Conditional current encoder must see only intrinsic flow features")
    dimension = len(detector.feature_names)
    if tuple(data["feature_names"]) != detector.feature_names + CONTEXT_FEATURES:
        raise ValueError("Expected current features followed by complete causal history")
    origin = json.loads((detector_path.parent / "registration.json").read_text())
    current_path = Path(origin["args"]["data"]).resolve()
    current_hash = sha256(current_path)
    if current_hash != detector.state["source_data_sha256"]:
        raise ValueError("Current encoder input was altered")
    with np.load(current_path) as original:
        for key in data:
            expected = data[key]
            if key == "features":
                expected = expected[:, :dimension]
            elif key == "feature_names":
                expected = expected[:dimension]
            if not np.array_equal(original[key], expected):
                raise ValueError(f"Conditional and current observations differ: {key}")
    control_registration = json.loads((control_path / "registration.json").read_text())
    if control_registration["data_sha256"] != current_hash:
        raise ValueError("Control decisions belong to different observations")
    if control_registration["detector_sha256"] != sha256(detector_path):
        raise ValueError("Control used a different current encoder")
    if control_registration["aggregation"] != "top3":
        raise ValueError("This matched experiment requires top3 control aggregation")
    stored = torch.load(control_path / "scores.pt", map_location="cpu", weights_only=True)
    target_minutes = stored["minutes"].numpy()
    roles = {key: value.numpy() for key, value in detector.state["source_role_indices"].items()}
    timestamp, label = data["observed_at"], data["label"]
    if any(np.any(label[ids] != 0) for ids in roles.values()):
        raise ValueError("A source role contains attacks")
    if any(timestamp[ids].max() >= target_minutes.min() * 60 for ids in roles.values()):
        raise ValueError("A source role crosses the first target query")
    if len(np.unique(np.concatenate(list(roles.values())))) != sum(map(len, roles.values())):
        raise ValueError("Source roles overlap")
    reference = roles["train"]
    reference = reference[np.argsort(data["signature"][reference])[: args.max_reference]]
    selected = roles["selection"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": str(data["dataset"]),
            "data_sha256": sha256(path),
            "detector_sha256": sha256(detector_path),
            "current_data_sha256": current_hash,
            "control_scores_sha256": sha256(control_path / "scores.pt"),
            "census_sha256": control_registration["census_sha256"],
            "fit": "frozen current encoder; joint support on exact source-train normals only",
            "score": "conditional current density; context marginal density cancels",
            "selection": "normal-only median/q95 scales; both 1 and8 modes reported",
            "query_labels": "original whole-table census reused from identical control windows",
            "calibration": "same held-out source minutes, strictly prior verified target normals",
        },
    )
    intrinsic = torch.from_numpy(data["features"][:, :dimension].copy())
    history = torch.from_numpy(data["features"][:, dimension:].copy())
    hmean = history[reference].mean(dim=0)
    hscale = history[reference].std(dim=0, unbiased=False).clamp_min(0.05)
    history = (history - hmean) / hscale
    current = detector.normalize_features(intrinsic)
    row_scores, fitted = {}, {}
    for modes in (1, 8):
        support = HistoryConditionalSupport.fit(current[reference], history[reference], modes)
        name = f"raw_conditional_m{modes}"
        row_scores[name] = scores(support, current, history).numpy()
        row_scores[f"raw_joint_marginal_m{modes}"] = scores(
            support, current, history, marginal=True
        ).numpy()
        fitted[name] = support.state_dict()
        print("fitted", name, "components", support.joint.components, flush=True)
    deep = {1: [], 8: []}
    marginal_deep = {1: [], 8: []}
    for model, item in zip(detector.models, detector.state["seed_models"], strict=True):
        embedded = torch.cat([model.layers[:-1](chunk) for chunk in current.split(8192)])
        embedded = (embedded - item["embedding_mean"]) / item["embedding_scale"]
        for modes in (1, 8):
            support = HistoryConditionalSupport.fit(embedded[reference], history[reference], modes)
            energy = scores(support, embedded, history)
            median, q95 = torch.quantile(
                energy[selected], torch.tensor([0.5, 0.95], dtype=torch.float64)
            )
            scale = max(float(q95 - median), 1e-6)
            deep[modes].append((energy - float(median)) / scale + 1)
            fitted[f"seed_{item['seed']}_m{modes}"] = {
                "support": support.state_dict(),
                "center": float(median),
                "scale": scale,
            }
            marginal = scores(support, embedded, history, marginal=True)
            marginal_median, marginal_q95 = torch.quantile(
                marginal[selected], torch.tensor([0.5, 0.95], dtype=torch.float64)
            )
            marginal_scale = max(float(marginal_q95 - marginal_median), 1e-6)
            marginal_deep[modes].append((marginal - float(marginal_median)) / marginal_scale + 1)
            fitted[f"seed_{item['seed']}_m{modes}"].update(
                {
                    "marginal_center": float(marginal_median),
                    "marginal_scale": marginal_scale,
                }
            )
            print("fitted seed", item["seed"], "modes", modes, flush=True)
    for modes, energies in deep.items():
        row_scores[f"deep_conditional_m{modes}_ensemble"] = torch.stack(energies).mean(0).numpy()
        row_scores[f"deep_joint_marginal_m{modes}_ensemble"] = (
            torch.stack(marginal_deep[modes]).mean(0).numpy()
        )
    torch.save(
        {
            "format_version": 1,
            "current_encoder": detector.state,
            "history_mean": hmean,
            "history_scale": hscale,
            "history_features": list(CONTEXT_FEATURES),
            "source_reference_indices": torch.from_numpy(reference.copy()),
            "conditional_supports": fitted,
        },
        output / "conditional_detector.pt",
    )
    restored = ConditionalCurrentDetector.load(output / "conditional_detector.pt")
    audit_rows = np.linspace(0, len(intrinsic) - 1, 128, dtype=np.int64)
    reproduced = restored.score(torch.from_numpy(data["features"][audit_rows]))
    replay_audit = {}
    for name, values in reproduced.items():
        expected = torch.from_numpy(row_scores[name][audit_rows])
        error = float(((values - expected).abs() / (1 + expected.abs())).max())
        if error > 1e-5:
            raise ValueError(f"Restored conditional inference differs: {name}, {error}")
        replay_audit[name] = {"max_scaled_error": error}
    minutes = timestamp // 60
    order = np.argsort(minutes, kind="stable")
    unique, starts, counts = np.unique(minutes[order], return_index=True, return_counts=True)
    groups = {
        int(t): order[start : start + count]
        for t, start, count in zip(unique, starts, counts, strict=True)
    }
    source_minutes = stored["methods"]["raw_knn"]["source_calibration_minutes"].numpy()
    membership = np.zeros(len(minutes), dtype=bool)
    membership[roles["calibration"]] = True
    for minute in source_minutes:
        if not membership[groups[int(minute)]].all():
            raise ValueError("A source calibration minute contains another role")
    archive = dict(stored["methods"])
    for name, values in row_scores.items():

        def aggregate(minute, energies=values):
            return float(np.sort(energies[groups[int(minute)]])[-3:].mean())

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
    replay = torch.load(output / "evaluation/decisions.pt", weights_only=True)
    for name, values in stored["methods"].items():
        for key in ("flags", "valid"):
            if not torch.equal(replay[name][key], values[key]):
                raise ValueError(f"Matched frozen control replay changed: {name}, {key}")
    report = json.loads((output / "evaluation/report.json").read_text())
    write_json(
        output / "report.json",
        {
            **report,
            "unit": "conditional_current_flow_whole_minute",
            "all_frozen_control_decisions_exact": True,
            "restored_inference_audit": replay_audit,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--buffer", type=int, default=39)
    parser.add_argument("--max-reference", type=int, default=20000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
