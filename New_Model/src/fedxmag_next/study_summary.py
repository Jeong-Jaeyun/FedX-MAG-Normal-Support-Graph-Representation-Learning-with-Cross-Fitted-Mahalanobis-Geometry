"""Summarize measured attempts and paired capture-cluster uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .detector import conformal_threshold
from .experiments import fuse, load_roles, summarize, write_json


def bootstrap(
    baseline: dict,
    candidate: dict,
    windows: dict,
    band: str,
    resamples: int = 2000,
) -> dict:
    roles = ("calibration", f"{band}_normal", f"{band}_attack")
    captures = sorted({w.metadata["session_key"] for role in roles for w in windows[role]})
    positions = {capture: index for index, capture in enumerate(captures)}
    members = {
        role: np.asarray([positions[w.metadata["session_key"]] for w in windows[role]])
        for role in roles
    }
    generator = np.random.default_rng(9187)
    differences = {"auroc": [], "tpr": [], "fpr": []}
    for _ in range(resamples):
        counts = np.bincount(
            generator.integers(len(captures), size=len(captures)), minlength=len(captures)
        )
        indices = {
            role: np.repeat(np.arange(len(members[role])), counts[members[role]]) for role in roles
        }
        if min(len(values) for values in indices.values()) < 19:
            continue
        results = []
        for scores in (baseline, candidate):
            cal, normal, attack = [scores[role][indices[role]] for role in roles]
            threshold = conformal_threshold(cal, 0.05)
            labels = np.concatenate((np.zeros(len(normal)), np.ones(len(attack))))
            values = np.concatenate((normal, attack))
            results.append(
                {
                    "auroc": float(roc_auc_score(labels, values)),
                    "tpr": float(np.mean(attack > threshold)),
                    "fpr": float(np.mean(normal > threshold)),
                }
            )
        for key in differences:
            differences[key].append(results[1][key] - results[0][key])
    return {
        key: {
            "mean_delta": float(np.mean(values)),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "resamples": len(values),
        }
        for key, values in differences.items()
    }


def run(args: argparse.Namespace) -> None:
    root, destination = Path(args.root).resolve(), Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    windows, _ = load_roles(root)
    runs = root / "New_Model/runs"
    attempts = {}
    for path in sorted(runs.glob("*/report.json")):
        attempts[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    seeds = [17, 29, 41]
    archives = [
        torch.load(runs / f"round03_dependency_seed{seed}/scores.pt", weights_only=True)
        for seed in seeds
    ]
    baseline = {
        role: values.numpy() for role, values in archives[0]["archived_raw_node_flow_knn"].items()
    }
    channels = [
        {role: value.numpy() for role, value in archive["flow_dependency_energy"].items()}
        for archive in archives
    ]
    average = {
        role: np.mean([channel[role] for channel in channels], axis=0) for role in channels[0]
    }
    ensemble = fuse([baseline, average])
    report = {
        "status": "ongoing_development_success_not_yet_verified",
        "completed_attempts": list(attempts),
        "ensemble_definition": (
            "Mean of selection-normal-scaled neural energies from seeds 17/29/41; "
            "max with fixed raw baseline"
        ),
        "baseline": summarize(baseline, windows),
        "ensemble": summarize(ensemble, windows),
        "paired_capture_bootstrap": {
            band: bootstrap(baseline, ensemble, windows, band, args.resamples)
            for band in ("test", "audit")
        },
        "limits": (
            "Repeated development; bootstrap conditional on frozen fits; "
            "captures, not windows, are resampled"
        ),
    }
    write_json(destination / "summary.json", report)
    torch.save(
        {
            "baseline": {r: torch.from_numpy(v) for r, v in baseline.items()},
            "ensemble": {r: torch.from_numpy(v) for r, v in ensemble.items()},
        },
        destination / "scores.pt",
    )
    lines = [
        "# Ongoing experiment results",
        "",
        "All rows are measured development results.",
        "",
        "| Run / method | Test AUROC | Test TPR / FPR | Audit AUROC | Audit TPR / FPR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, attempt in attempts.items():
        for method, result in attempt.get("methods", {}).items():
            if "test" not in result:
                continue
            test, audit = result["test"], result["audit"]
            lines.append(
                f"| {name} / {method} | {test['auroc']:.3f} | "
                f"{test['tpr']:.1%} / {test['fpr']:.1%} | {audit['auroc']:.3f} | "
                f"{audit['tpr']:.1%} / {audit['fpr']:.1%} |"
            )
    lines.extend(
        [
            "",
            "## Three-seed neural ensemble",
            "",
            "```json",
            json.dumps(
                {
                    "metrics": {
                        b: {k: v[k] for k in ("auroc", "tpr", "fpr")}
                        for b, v in report["ensemble"].items()
                    },
                    "paired_capture_bootstrap": report["paired_capture_bootstrap"],
                },
                indent=2,
            ),
            "```",
            "",
            "Overall success remains unproven; failed capture/domain gates remain in scope.",
            "",
        ]
    )
    (destination / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report["paired_capture_bootstrap"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
