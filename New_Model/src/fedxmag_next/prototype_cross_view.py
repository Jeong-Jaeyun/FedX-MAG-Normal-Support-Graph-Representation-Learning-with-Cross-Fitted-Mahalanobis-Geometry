"""Use raw normal-prototype distances with unchanged deep anomaly energies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .experiments import sha256, write_json
from .hierarchical_experiment import minute_groups
from .prototype_guard import SCOPES
from .prototype_temporal import frozen_thresholds, replay_path
from .service_risk_keys import PROTOCOL_COLUMNS, service_keys
from .window_replay import run as evaluate_windows


def run(args):
    torch.set_num_threads(args.workers)
    service_mode = getattr(args, "intervention", "raw_normal_view") == "service_keys"
    origin, control, output = map(
        lambda p: Path(p).resolve(), (args.run, args.control, args.output)
    )
    registration = json.loads((origin / "registration.json").read_text())
    risk_root = Path(registration["args"]["run"]).resolve()
    risk_registration = json.loads((risk_root / "registration.json").read_text())
    data_path, key_path = map(
        lambda k: Path(risk_registration["args"][k]).resolve(), ("data", "keys")
    )
    if (
        sha256(data_path) != registration["data_sha256"]
        or sha256(key_path) != registration["keys_sha256"]
    ):
        raise ValueError("Cross-view observations or key metadata changed")
    if sha256(risk_root / "row_scores.pt") != registration["row_cache_sha256"]:
        raise ValueError("Cross-view frozen row-energy cache changed")
    cache = torch.load(risk_root / "row_scores.pt", weights_only=True)
    distances = torch.load(origin / "prototype_distances.pt", weights_only=True)
    if not torch.equal(cache["combined_indices"], distances["combined_indices"]):
        raise ValueError("Cross-view normal distance rows differ from frozen energy rows")
    combined = cache["combined_indices"].numpy()
    with np.load(data_path) as payload:
        times, signature = payload["observed_at"][combined], payload["signature"]
        if service_mode:
            names = payload["feature_names"].tolist()
            columns = [names.index(k) for k in (*PROTOCOL_COLUMNS, "available_dst_port")]
            metadata = payload["features"][:, columns][combined]
    with np.load(key_path) as payload:
        if not np.array_equal(signature, payload["signature"]):
            raise ValueError("Cross-view endpoint equality keys are misaligned")
        keys = {k: payload[k][combined] for k in ("source", "destination")}
        if service_mode:
            keys["port"] = payload["port"][combined]
    routed_services = (
        {scope: service_keys(keys, metadata[:, :4], metadata[:, 4], scope) for scope in SCOPES}
        if service_mode
        else {}
    )
    groups = minute_groups(times // 60)
    original = torch.load(origin / "scores.pt", weights_only=True)
    output.mkdir(parents=True, exist_ok=False)
    common = {
        "args": vars(args),
        "dataset": registration["dataset"],
        "data_sha256": registration["data_sha256"],
        "source_detector_sha256": registration["detector_sha256"],
        "prototype_model_sha256": sha256(origin / "prototype_risk_detector.pt"),
        "prototype_distances_sha256": sha256(origin / "prototype_distances.pt"),
        "row_cache_sha256": registration["row_cache_sha256"],
        "normal_view": "raw_and_deep" if service_mode else "raw",
        "anomaly_view": "unchanged deep1/8",
        "change": (
            "only risk keys refined by observed destination port/protocol/availability"
            if service_mode
            else "only normal-filter/gate distance view changes; raw q95 radii frozen"
        ),
        "intervention": "service_keys" if service_mode else "raw_normal_view",
        "aggregation": registration["aggregation"],
        "feedback_modes": ["original frozen energy", "candidate energy"],
        "calibration_gate": "fixed original-feedback base gate for both final feedback modes",
        "source_precision_audit": (
            "recomputed matched deep window energies must equal parent exactly"
        ),
        "normal_admission": "unchanged actual verified-normal cache, prediction before admission",
    }
    write_json(output / "registration.json", common)
    torch.save(
        {
            "format_version": 1,
            "normal_view": common["normal_view"],
            "parent_model": str(origin / "prototype_risk_detector.pt"),
            "parent_model_sha256": common["prototype_model_sha256"],
            "protocol": common,
        },
        output / "cross_view_detector.pt",
    )
    for capacity in (39, 50):
        source = control if capacity == 39 else control / "capacity50"
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir(exist_ok=False)
            write_json(destination / "registration.json", {**common, "capacity": capacity})
        old = torch.load(source / "scores.pt", weights_only=True)
        for field in ("minutes", "labels"):
            if not torch.equal(old[field], original[field]):
                raise ValueError(f"Cross-view control changed windows/truth: {field}")
        if old["types"] != original["types"]:
            raise ValueError("Cross-view control changed attack-type census")
        minutes, labels = old["minutes"].numpy(), old["labels"].numpy()
        archive, diagnostics, errors, expected_flags = dict(old["methods"]), {}, {}, {}
        containments, refinements = [], {}
        checkpoint_folder = destination / "checkpoints"
        checkpoint_folder.mkdir()
        for name, tensor in cache["row_scores"].items():
            if not name.startswith("deep_"):
                continue
            baseline = old["methods"][name]
            thresholds = frozen_thresholds(
                baseline["energy"].numpy(),
                baseline["observed"].numpy(),
                labels,
                baseline["source_calibration"].tolist(),
                capacity,
            )
            for scope in SCOPES:
                designs = (
                    [
                        (
                            "prototype_risk_fixed",
                            view,
                            view,
                            ("rawview_" if view == "raw" else "")
                            + "service_continuous_prototype_risk_fixed",
                        )
                        for view in ("deep", "raw")
                    ]
                    if service_mode
                    else [
                        (path, "deep", "raw", f"rawview_continuous_{path}")
                        for path in ("prototype", "prototype_risk_fixed")
                    ]
                )
                for path, parent_view, child_view, tag in designs:
                    prefix = "rawview_" if parent_view == "raw" else ""
                    matched_name = f"{name}_{prefix}continuous_{path}_{scope}"
                    matched, matched_diagnostics = replay_path(
                        tensor.numpy(),
                        cache["row_reference"][name].numpy(),
                        distances["ratios"][scope][parent_view].numpy(),
                        minutes,
                        groups,
                        keys,
                        times,
                        registration["aggregation"],
                        thresholds,
                        path,
                        scope,
                        checkpoint_folder / f"matched_{matched_name}.pt",
                        continuous_filter=True,
                    )
                    expected = old["methods"][matched_name]["energy"].numpy()
                    if not np.array_equal(matched, expected):
                        error = float(np.max(np.abs(matched - expected)))
                        raise ValueError(
                            f"Sorted-reference matched-view replay differs: {matched_name}/{error}"
                        )
                    errors[matched_name] = 0.0
                    head = f"{name}_{tag}_{scope}"
                    values, diagnostics[head] = replay_path(
                        tensor.numpy(),
                        cache["row_reference"][name].numpy(),
                        distances["ratios"][scope][child_view].numpy(),
                        minutes,
                        groups,
                        keys,
                        times,
                        registration["aggregation"],
                        thresholds,
                        path,
                        scope,
                        checkpoint_folder / f"{head}.pt",
                        continuous_filter=True,
                        risk_key_values=routed_services[scope] if service_mode else None,
                    )
                    if service_mode:
                        refinements[head] = {
                            "max_positive_parent_energy_difference": float(
                                np.maximum(values - matched, 0).max()
                            ),
                            "parent_maximum_risk_keys": matched_diagnostics["maximum_risk_keys"],
                            "child_maximum_risk_keys": diagnostics[head]["maximum_risk_keys"],
                        }
                        containments.append((f"{name}_{prefix}continuous_prototype_{scope}", head))
                    for feedback in ("frozen", "candidate"):
                        alias = (
                            head
                            if feedback == "frozen"
                            else head.replace("_continuous_", "_adaptive_continuous_", 1)
                        )
                        archive[alias] = {
                            "energy": torch.from_numpy(values.copy()),
                            "observed": baseline["observed"].clone(),
                            "source_calibration": baseline["source_calibration"].clone(),
                            "source_calibration_minutes": baseline[
                                "source_calibration_minutes"
                            ].clone(),
                            "feedback_energy": baseline["energy"].clone()
                            if feedback == "frozen"
                            else torch.from_numpy(values.copy()),
                        }
                    expected_flags[head] = (values > thresholds) & baseline["observed"].numpy()
        torch.save({**old, "methods": archive}, destination / "scores.pt")
        evaluate_windows(
            argparse.Namespace(
                run=str(destination),
                output=str(destination / "evaluation"),
                buffer=capacity,
                fpr=0.05,
                history_policy="continuous",
            )
        )
        actual = torch.load(destination / "evaluation/decisions.pt", weights_only=True)
        parent = torch.load(source / "evaluation/decisions.pt", weights_only=True)
        for name, item in parent.items():
            if any(not torch.equal(actual[name][k], v) for k, v in item.items()):
                raise ValueError(f"Cross-view parent decision changed: {name}")
        for name, flags in expected_flags.items():
            if not np.array_equal(actual[name]["flags"].numpy(), flags):
                raise ValueError("Cross-view frozen-feedback replay changed fixed thresholds")
        if not service_mode:
            for name in cache["row_scores"]:
                if not name.startswith("deep_"):
                    continue
                for scope in SCOPES:
                    prototype = actual[f"{name}_rawview_continuous_prototype_{scope}"]["flags"]
                    joint = actual[f"{name}_rawview_continuous_prototype_risk_fixed_{scope}"][
                        "flags"
                    ]
                    if torch.any(prototype & ~actual[name]["flags"]) or torch.any(
                        prototype & ~joint
                    ):
                        raise ValueError("Cross-view normal-filter alarm containment failed")
        for prototype, joint in containments:
            if torch.any(actual[prototype]["flags"] & ~actual[joint]["flags"]):
                raise ValueError("Service-specific risk lost a prototype-only alarm")
        report = json.loads((destination / "evaluation/report.json").read_text())
        write_json(
            destination / "report.json",
            {
                **report,
                "all_parent_decisions_exact": True,
                "matched_view_recomputed_exact": True,
                "matched_view_energy_errors": errors,
                "normal_filter_alarm_subset_exact": True,
                "path_diagnostics": diagnostics,
                "service_refinement_diagnostics": refinements,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "control", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--intervention", choices=("raw_normal_view", "service_keys"), default="raw_normal_view"
    )
    run(parser.parse_args())
