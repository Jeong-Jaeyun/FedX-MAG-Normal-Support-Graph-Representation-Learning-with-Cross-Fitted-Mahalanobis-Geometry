"""Independently replay all H2/H3 decisions from a six-run manifest.

--runs is a JSON list of {hypothesis: h2|h3, dataset: cic|xiiotid|ton,
run: absolute directory}. --smoke requires six explicitly marked smoke runs;
without it, every full run must match the original complete H1 target census.
No trained model or target-dependent threshold selection is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch

POLICIES = ("fixed", "delayed_10pct", "immediate")


def registered_methods(hypothesis, smoke=False):
    """Independent expected set: absence from all archives must not pass audit."""
    if hypothesis == "h2":
        names = {"raw_m8_reference"}
        if not smoke:
            names.add("frozen_dnn20k_m8_reference")
        for corruption, kind, head in product(
            ("original", "conditional"),
            ("quantum", "quantum_noent", "mlp32", "mlp_budget", "full_mlp64"),
            ("scalar", "support"),
        ):
            branch = f"{corruption}_{kind}_{head}"
            names.update((branch, "raw_plus_" + branch))
        return names
    if hypothesis == "h3":
        return {
            f"{kind}_m{mode}"
            for kind, mode in product(
                (
                    "ar_ridge",
                    "delay_support",
                    "esn",
                    "gru",
                    "quantum",
                    "quantum_noent",
                    "quantum_zero_history",
                    "quantum_shuffled",
                ),
                (1, 8),
            )
        }
    raise ValueError("Unknown registered hypothesis")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vector(value, name, length=None):
    result = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    require(result.ndim == 1, f"{name}: expected vector")
    require(np.isfinite(result).all(), f"{name}: nonfinite values")
    if length is not None:
        require(len(result) == length, f"{name}: incomplete coverage")
    return result


def independent_replay(energy, source, minutes, labels, policy):
    """Sorted-list implementation independent of the production replay code."""
    require(policy in POLICIES, "Unknown policy")
    require(len(source) >= 19, "Insufficient normal calibration")
    history, pending = list(source[-50:]), []
    chosen = np.random.default_rng(731).random(len(minutes)) < 0.1
    flags, thresholds = [], []
    for index, minute in enumerate(minutes):
        now = (int(minute) + 1) * 60
        history += [value for arrival, value in pending if arrival <= now]
        pending = [(arrival, value) for arrival, value in pending if arrival > now]
        history = history[-50:]
        threshold = sorted(history)[math.ceil((len(history) + 1) * 0.95) - 1]
        thresholds.append(threshold)
        flags.append(energy[index] > threshold)
        # Current truth can affect only later decisions, never its own threshold.
        if labels[index] == 0 and policy == "immediate":
            history.append(float(energy[index]))
        elif labels[index] == 0 and policy == "delayed_10pct" and chosen[index]:
            pending.append((now + 300, float(energy[index])))
    return np.asarray(flags, dtype=bool), np.asarray(thresholds, dtype=np.float64)


def paired_comparison(first, second, labels, minutes, bootstrap=True):
    result = {
        "added_tp": int((first & ~second & (labels == 1)).sum()),
        "lost_tp": int((~first & second & (labels == 1)).sum()),
        "added_fp": int((first & ~second & (labels == 0)).sum()),
        "removed_fp": int((~first & second & (labels == 0)).sum()),
    }
    blocks, inverse = np.unique(minutes // 60, return_inverse=True)
    if bootstrap:
        weights = np.random.default_rng(731).multinomial(
            len(blocks), np.full(len(blocks), 1 / len(blocks)), size=2000
        )
        result["observed_hour_blocks"] = len(blocks)
        result["bootstrap_samples"] = 2000
    for label, name in ((1, "tpr"), (0, "fpr")):
        selected = labels == label
        changes = first[selected].astype(float) - second[selected]
        entry = {"difference": float(changes.mean())}
        if bootstrap:
            numerator = np.bincount(inverse[selected], weights=changes, minlength=len(blocks))
            denominator = np.bincount(inverse[selected], minlength=len(blocks))
            den = weights @ denominator
            samples = (weights @ numerator)[den > 0] / den[den > 0]
            entry["hour_bootstrap_95"] = np.quantile(samples, [0.025, 0.975]).tolist()
        result[name] = entry
    return result


def audit_run(specification, root, smoke=False):
    run = Path(specification["run"])
    hypothesis, dataset = specification["hypothesis"], specification["dataset"]
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    registration = json.loads((run / "registration.json").read_text(encoding="utf-8"))
    require(report.get("status") == "completed", f"{run}: incomplete report")
    require(report.get("dataset") == dataset, f"{run}: dataset differs")
    if hypothesis == "h2":
        require(report.get("smoke") is smoke, f"{run}: H2 smoke scope mismatch")
        require(registration.get("smoke_only") is smoke, f"{run}: registration scope mismatch")
        if not smoke:
            require(report.get("full_target_rows_and_windows") is True, "H2 census unconfirmed")
    else:
        expected = "integration_smoke" if smoke else "full_prepared_target_census"
        require(report.get("scope") == expected, f"{run}: H3 scope mismatch")
        require(registration.get("scope") == expected, f"{run}: registration scope mismatch")
    scores = torch.load(run / "scores.pt", weights_only=True)
    decisions = torch.load(run / "decisions.pt", weights_only=True)
    minutes = vector(scores["minutes"], "minutes")
    labels = vector(scores["labels"], "labels", len(minutes))
    calibration_minutes = vector(scores["source_calibration_minutes"], "calibration_minutes")
    require(len(minutes) > 1 and np.all(np.diff(minutes) > 0), "Nonincreasing target chronology")
    require(
        len(calibration_minutes) >= 19 and np.all(np.diff(calibration_minutes) > 0),
        "Nonincreasing or insufficient calibration chronology",
    )
    require(calibration_minutes.max() < minutes.min(), "Future calibration")
    require(np.isin(labels, (0, 1)).all() and len(np.unique(labels)) == 2, "Invalid target labels")
    require(np.equal(minutes, minutes.astype(np.int64)).all(), "Noninteger target minutes")
    require(len(scores["types"]) == len(minutes), "Incomplete subtype coverage")
    if not smoke:
        parent = Path(root) / f"New_Model/runs/round233_h1_{dataset}_pilot_v2/scores.pt"
        reference = torch.load(parent, weights_only=True)
        require(np.array_equal(minutes, reference["minutes"].numpy()), "Target census differs")
        require(np.array_equal(labels, reference["labels"].numpy()), "Target labels differ")
        require(scores["types"] == reference["types"], "Target subtypes differ")
    methods = scores["methods"]
    require(
        set(methods) == registered_methods(hypothesis, smoke),
        "Preregistered method coverage differs",
    )
    require(bool(methods) and set(methods) == set(report["methods"]), "Methods missing from report")
    require(
        set(decisions) == {n + "/" + p for n in methods for p in POLICIES},
        "Decision archive missing or adding methods/policies",
    )
    for name, values in methods.items():
        energy = vector(values["energy"], name + "/energy", len(minutes))
        source = vector(values["source_calibration"], name + "/source", len(calibration_minutes))
        require(set(report["methods"][name]) == set(POLICIES), "Reported policy coverage differs")
        for policy in POLICIES:
            flags, thresholds = independent_replay(energy, source, minutes, labels, policy)
            saved = decisions[name + "/" + policy]
            stored_flags = vector(saved["flags"], name + "/flags", len(minutes))
            stored_thresholds = vector(saved["thresholds"], name + "/thresholds", len(minutes))
            require(np.array_equal(flags, stored_flags), f"{name}/{policy}: flags differ")
            require(
                np.array_equal(thresholds, stored_thresholds), f"{name}/{policy}: thresholds differ"
            )
            metrics = report["methods"][name][policy]
            tp, fp = int(flags[labels == 1].sum()), int(flags[labels == 0].sum())
            require(metrics["tp"] == tp and metrics["fp"] == fp, f"{name}: counts differ")
            attacks, normals = int((labels == 1).sum()), int((labels == 0).sum())
            require(
                metrics["attack_windows"] == attacks and metrics["normal_windows"] == normals,
                f"{name}: class coverage differs",
            )
            require(metrics["coverage"] == 1.0, f"{name}: incomplete reported coverage")
            require(
                metrics["tpr"] == tp / attacks and metrics["fpr"] == fp / normals,
                f"{name}: rates differ",
            )
    source_artifact = torch.load(run / "source_adapter.pt", weights_only=True)
    role_key = "source_role_indices" if hypothesis == "h2" else "observation_ids"
    roles = {name: vector(ids, name) for name, ids in source_artifact[role_key].items()}
    require(len(roles) >= 3, "Missing source roles")
    for first, second in combinations(roles, 2):
        require(
            not np.intersect1d(roles[first], roles[second]).size,
            f"Overlapping source observations: {first}/{second}",
        )
    comparisons = {}
    for name in methods:
        if "quantum_noent" in name or "quantum" not in name:
            continue
        reference = name.replace("quantum", "quantum_noent", 1)
        if reference not in methods:
            continue
        comparisons[name + "_vs_" + reference] = {
            policy: paired_comparison(
                decisions[name + "/" + policy]["flags"].numpy(),
                decisions[reference + "/" + policy]["flags"].numpy(),
                labels,
                minutes,
                bootstrap=not smoke,
            )
            for policy in POLICIES
        }
    filenames = (
        "report.json",
        "registration.json",
        "scores.pt",
        "decisions.pt",
        "source_adapter.pt",
    )
    return {
        "hypothesis": hypothesis,
        "dataset": dataset,
        "run": str(run),
        "scope": "integration_smoke" if smoke else "full_prepared_target_census",
        "target_windows": len(minutes),
        "attack_windows": int((labels == 1).sum()),
        "normal_windows": int((labels == 0).sum()),
        "methods_checked": len(methods),
        "policy_conditions_checked": len(methods) * len(POLICIES),
        "all_decisions_thresholds_counts_rates_exact": True,
        "all_source_observation_roles_disjoint": True,
        "full_target_census_matches_h1": not smoke,
        "artifacts": {
            name: {"path": str(run / name), "sha256": digest(run / name)} for name in filenames
        },
        "methods": report["methods"],
        "representation_audits": report.get("audits", {}),
        "training_entries": len(report.get("training", {})),
        "full_training_and_audits_report": str(run / "report.json"),
        "paired_quantum_noent": comparisons,
        "elapsed_seconds": report.get("elapsed_seconds"),
    }


def render_tables(result):
    lines = [
        "# H2/H3 independently audited results",
        "",
        "Each cell is TP/FP (TPR% / FPR%). All listed policies use the same registered "
        "normal-only calibration rule. Actual FPR is not fixed by target labels.",
        "",
        "Scope: " + result["scope"] + ". CPU simulation; no QPU execution or quantum advantage "
        "claim. H2 is a static representation/head/corruption comparison; H3 is a separately "
        "purged temporal prediction-residual comparison. Existing full adaptive detector "
        "results are not matched baselines. These are reused development data.",
        "",
    ]
    if result["smoke"]:
        lines += ["**Integration smoke only: these values are not full accuracy estimates.**", ""]
    for run in result["runs"]:
        lines += [
            f"## {run['hypothesis'].upper()} / {run['dataset']}",
            "",
            f"Attack windows: {run['attack_windows']}; normal windows: {run['normal_windows']}.",
            "",
            f"Full training/audit record: [{run['run']}/report.json]({run['run']}/report.json)",
            "",
            "| Method | Fixed | Delayed 10%, 300 s | Immediate |",
            "|---|---:|---:|---:|",
        ]
        for name, policies in run["methods"].items():
            cells = [name]
            for policy in POLICIES:
                m = policies[policy]
                cells.append(f"{m['tp']}/{m['fp']} ({100 * m['tpr']:.2f} / {100 * m['fpr']:.2f})")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def write_new_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def run(args):
    root = Path(args.root).resolve()
    manifests = Path(args.runs)
    specifications = json.loads(manifests.read_text(encoding="utf-8"))
    require(isinstance(specifications, list), "--runs must contain a JSON list")
    expected = set(product(("h2", "h3"), ("cic", "xiiotid", "ton")))
    actual = [(s["hypothesis"], s["dataset"]) for s in specifications]
    require(len(actual) == 6 and set(actual) == expected, "Exactly all six H2/H3 datasets required")
    require(all(Path(s["run"]).is_absolute() for s in specifications), "Run paths must be absolute")
    require(
        len({str(Path(s["run"]).resolve()).casefold() for s in specifications}) == 6,
        "Run paths must be distinct",
    )
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "running",
        "smoke": args.smoke,
        "scope": "integration_smoke" if args.smoke else "full_prepared_target_census",
        "claim": "reused development data; CPU simulation; no quantum advantage claim",
        "analysis_source_sha256": digest(__file__),
        "run_manifest_sha256": digest(manifests),
        "bootstrap_scope": "descriptive observed-hour paired resampling of frozen decisions; "
        "not independent-site inference, not feedback replay under resampling, "
        "not adjusted for multiple comparisons",
        "runs": [],
    }
    try:
        for specification in specifications:
            checked = audit_run(specification, root, smoke=args.smoke)
            result["runs"].append(checked)
            write_new_json(output / f"progress_{len(result['runs']):02d}.json", checked)
            print(
                checked["hypothesis"],
                checked["dataset"],
                "audit passed",
                checked["policy_conditions_checked"],
                flush=True,
            )
        result["status"] = "completed"
        result["total_policy_conditions_checked"] = sum(
            r["policy_conditions_checked"] for r in result["runs"]
        )
        with (output / "all_methods.md").open("x", encoding="utf-8") as handle:
            handle.write(render_tables(result))
        write_new_json(output / "report.json", result)
        print("AUDIT COMPLETE", output / "report.json", flush=True)
    except Exception as error:
        write_new_json(
            output / "failure.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "completed_runs": len(result["runs"]),
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
