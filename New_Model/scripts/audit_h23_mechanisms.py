"""Independently verify every saved synthetic H2/H3 mechanism decision.

No predictor is trained or rescored here. Normal PCA statistics are recomputed
from saved train observations to verify actual adapters. Negative-control
outcomes are never used to discard a seed or declare an implementation bug.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
import torch

from fedxmag_next.h1_classical_features import SourcePCA

REPLAY_SOURCE = Path(__file__).with_name("audit_h23_results.py")
_SPEC = importlib.util.spec_from_file_location("_mechanism_replay_core", REPLAY_SOURCE)
_REPLAY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REPLAY)
digest, require, vector = _REPLAY.digest, _REPLAY.require, _REPLAY.vector
independent_replay = _REPLAY.independent_replay

POLICIES = ("fixed", "delayed_10pct", "immediate")
DOMAINS = {
    "d8_m1": {"dimension": 8, "modes": 1, "noise": 0.05, "missing": 0.0},
    "d12_m2": {"dimension": 12, "modes": 2, "noise": 0.15, "missing": 0.05},
    "d16_m2": {"dimension": 16, "modes": 2, "noise": 0.30, "missing": 0.15},
}
FAMILIES = {
    "h2": ("parity_flip", "relationship_reverse", "normal_drift", "identical_labels"),
    "h3": ("order_reverse", "normal_drift", "identical_labels"),
}
ROLES = ("train", "selection", "support", "calibration")
H2_ROLES = ("train", "selection", "diagnostic_reference", "calibration")
H2_KINDS = ("quantum", "quantum_noent", "mlp32", "mlp_budget", "full_mlp64")
H3_KINDS = (
    "ar_ridge",
    "delay_support",
    "esn",
    "gru",
    "quantum",
    "quantum_noent",
    "quantum_zero_history",
    "quantum_shuffled",
)


def expected_methods(hypothesis):
    if hypothesis == "h2":
        return {
            f"{corruption}_{kind}_{head}"
            for corruption, kind, head in product(
                ("original", "conditional"), H2_KINDS, ("scalar", "support")
            )
        }
    require(hypothesis == "h3", "Unknown hypothesis")
    return {f"{kind}_m{mode}" for kind, mode in product(H3_KINDS, (1, 8))}


def expected_tasks():
    return {
        f"{hypothesis}/{domain}/{family}"
        for hypothesis in FAMILIES
        for domain in DOMAINS
        for family in FAMILIES[hypothesis]
    }


def expected_counts(hypothesis, smoke):
    if hypothesis == "h2":
        return (64, 32, 32, 32, 64) if smoke else (4096, 1024, 1024, 256, 1024)
    return (64, 32, 32, 32) if smoke else (256, 128, 128, 128)


def finite_matrix(value, dimension, name):
    result = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    require(result.ndim == 2 and result.shape[1] == 2 * dimension, f"{name}: input shape differs")
    require(np.isfinite(result).all(), f"{name}: nonfinite observation")
    availability = result[:, dimension:]
    require(np.isin(availability, (0, 1)).all(), f"{name}: nonbinary availability")
    require(
        np.all(result[:, :dimension][availability == 0] == 0), f"{name}: unmasked missing value"
    )
    return result


def verify_adapter(reference, actual, name, tolerance=1e-8):
    """Check the actual saved transform against a train-only recomputation.

    This deliberately rejects a different eigenspace basis as well as incorrect
    statistics. Any degenerate-basis exception must be investigated explicitly,
    rather than silently weakening source-role validation.
    """
    expected = reference.state_dict()
    require(set(actual) == set(expected), f"{name}: PCA state fields differ")
    errors = {}
    for field, value in expected.items():
        observed = torch.as_tensor(actual[field], dtype=torch.float64)
        require(
            observed.shape == value.shape and torch.isfinite(observed).all(),
            f"{name}: PCA {field} shape/values invalid",
        )
        error = float((observed - value).abs().max())
        require(error <= tolerance, f"{name}: PCA {field} differs from saved train fit: {error}")
        errors[field] = error
    return errors


def audit_source(folder, hypothesis, domain, entry, smoke):
    counts = expected_counts(hypothesis, smoke)
    require(tuple(entry["counts"]) == counts, f"{domain}/{hypothesis}: source budget differs")
    require(entry.get("observation_roles_disjoint") is True, "Source separation unconfirmed")
    draw_path = folder / "draws.pt"
    draws = torch.load(draw_path, weights_only=True)
    roles = H2_ROLES if hypothesis == "h2" else ROLES
    require(set(draws["observation_ids"]) == set(roles), "Missing source observation role")
    ids = {role: vector(draws["observation_ids"][role], role) for role in roles}
    for role, values in ids.items():
        require(
            np.equal(values, values.astype(np.int64)).all() and (values >= 0).all(),
            f"{role}: invalid observation IDs",
        )
        require(len(values) > 0 and np.all(np.diff(values) > 0), f"{role}: duplicate/unsorted IDs")
    for first, second in combinations(roles, 2):
        require(
            not np.intersect1d(ids[first], ids[second]).size,
            f"Source observations overlap: {first}/{second}",
        )
    dimension = DOMAINS[domain]["dimension"]
    if hypothesis == "h2":
        boundaries = np.cumsum((0, *counts[:4]))
        for i, role in enumerate(roles):
            raw = finite_matrix(draws[role], dimension, role)
            require(len(raw) == counts[i], f"{role}: incomplete source draw")
            require(
                np.array_equal(ids[role], np.arange(boundaries[i], boundaries[i + 1])),
                f"{role}: observation IDs do not identify the stored source rows",
            )
        require(
            draws["role_generator_seeds"]
            == {
                "train": 1001,
                "selection": 1002,
                "diagnostic_reference": 1003,
                "calibration": 1004,
            },
            "Source draw seeds differ",
        )
        adapter_fit = draws["train"]
    else:
        source = finite_matrix(draws["source"], dimension, "source")
        require(set(draws["role_endpoints"]) == set(ROLES), "Temporal endpoint roles missing")
        endpoints = {role: vector(draws["role_endpoints"][role], role) for role in ROLES}
        for i, role in enumerate(ROLES):
            query = endpoints[role]
            require(
                len(query) == counts[i] and np.all(np.diff(query) > 0),
                f"{role}: incomplete/unsorted temporal endpoints",
            )
            require(
                np.equal(query, query.astype(np.int64)).all()
                and query.min() >= 8
                and query.max() < len(source),
                "Invalid source history",
            )
            owned = np.unique(query[:, None] - np.arange(9))
            require(np.array_equal(ids[role], owned), f"{role}: history observation IDs differ")
        for first, second in zip(ROLES[:-1], ROLES[1:], strict=True):
            require(
                endpoints[second].min() - endpoints[first].max() >= 14,
                "Temporal endpoint purge shorter than13 unused minutes",
            )
        require(
            np.array_equal(vector(draws["adapter_fit_observations"], "adapter_fit"), ids["train"]),
            "Temporal PCA fitted outside the train observation role",
        )
        adapter_fit = draws["source"][torch.as_tensor(ids["train"], dtype=torch.long)]
    reference_adapter = SourcePCA.fit(adapter_fit)
    adapter_errors = {key: 0.0 for key in reference_adapter.state_dict()}
    adapters_checked = 0

    def check_adapter(actual, name):
        nonlocal adapters_checked
        errors = verify_adapter(reference_adapter, actual, name)
        for field, value in errors.items():
            adapter_errors[field] = max(adapter_errors[field], value)
        adapters_checked += 1

    require(set(draws["tasks"]) == set(FAMILIES[hypothesis]), "Source draw task coverage differs")
    n = counts[4] if hypothesis == "h2" else 64 if smoke else 1024
    for family, values in draws["tasks"].items():
        raw = finite_matrix(values["features"], dimension, family)
        require(len(raw) == n + (8 if hypothesis == "h3" else 0), "Task observation count differs")
        labels = vector(values["labels"], family + "/labels", n)
        require(np.isin(labels, (0, 1)).all(), "Invalid generated labels")
        if family == "normal_drift":
            require(not labels.any(), "Normal drift contains attack labels")
    seeds = (17,) if smoke else (17, 29, 41)
    model_keys = (
        {
            f"{corruption}/{kind}/{seed}"
            for corruption, kind, seed in product(("original", "conditional"), H2_KINDS, seeds)
        }
        if hypothesis == "h2"
        else {f"{kind}/{seed}" for kind, seed in product(H3_KINDS, seeds)}
    )
    require(set(entry["models"]) == model_keys, "Source-model/seed coverage differs")
    models, states = {}, {}
    for name, model in entry["models"].items():
        model_path = Path(model["path"])
        require(model_path.resolve().parent == folder.resolve(), "Model points outside domain role")
        require(model.get("restore_exact") is True, "Source model restore unconfirmed")
        require(digest(model_path) == model["sha256"], f"Changed source model: {name}")
        state = torch.load(model_path, weights_only=True)
        check_adapter(state["adapter"], name)
        if hypothesis == "h2":
            require(
                state.get("support_fit_role") == "train"
                and state.get("support_fit_rows") == counts[0],
                "H2 support must use the train role",
            )
            require(state.get("feature_sd_floor") == 1e-6, "H2 feature SD floor differs")
            geometries = [state["geometry"]]
            expected_rows = counts[0]
        else:
            geometries = [state["model"]["supports"][str(mode)]["geometry"] for mode in (1, 8)]
            expected_rows = counts[2]
        for geometry in geometries:
            require(
                int(vector(geometry["counts"], "geometry_counts").sum()) == expected_rows,
                "Saved geometry support-fit budget differs",
            )
        states[name] = state
        models[name] = {"path": str(model_path), "sha256": model["sha256"]}
    ensembles = {}
    if hypothesis == "h2":
        expected = {
            f"{corruption}/{kind}"
            for corruption, kind in product(("original", "conditional"), H2_KINDS)
        }
        require(set(entry["ensemble_audits"]) == expected, "H2 ensemble coverage differs")
        for name, spec in entry["ensemble_audits"].items():
            path = Path(spec["path"])
            require(
                path.resolve().parent == folder.resolve() and digest(path) == spec["sha256"],
                "H2 ensemble artifact differs",
            )
            require(
                set(spec["scaled_errors"]) == {"scalar", "support"}
                and all(np.isfinite(v) and 0 <= v <= 1e-7 for v in spec["scaled_errors"].values()),
                "H2 ensemble restoration failed",
            )
            state = torch.load(path, weights_only=True)
            require(
                set(state["ensemble_units"]) == {"scalar", "support"}, "H2 ensemble units missing"
            )
            require(len(state["seeds"]) == len(seeds), "H2 ensemble seed coverage differs")
            for seed, saved in zip(seeds, state["seeds"], strict=True):
                check_adapter(saved["adapter"], f"ensemble {name}/{seed}")
                equal_tensor_archives(states[f"{name}/{seed}"], saved, "ensemble_seed")
            for unit in state["ensemble_units"].values():
                require(
                    np.isfinite(unit["center"])
                    and np.isfinite(unit["scale"])
                    and unit["scale"] >= 1e-6,
                    "Invalid H2 ensemble units",
                )
            ensembles[name] = {"path": str(path), "sha256": spec["sha256"]}
    return draws, {
        "draws": {"path": str(draw_path), "sha256": digest(draw_path)},
        "models": models,
        "source_counts": dict(zip(roles, counts[:4], strict=True)),
        "observation_counts": {name: len(values) for name, values in ids.items()},
        "source_observation_roles_disjoint": True,
        "history_observation_membership_exact": True if hypothesis == "h3" else None,
        "adapter_train_only_verified": True,
        "adapter_train_audit": {
            "reference_fit_role": "train",
            "reference_fit_observations": len(adapter_fit),
            "reference_method": "SourcePCA.fit actual saved train observations only",
            "adapters_checked": adapters_checked,
            "absolute_tolerance": 1e-8,
            "max_absolute_errors": adapter_errors,
            "degenerate_basis_exception_used": False,
        },
        "support_fit_budget_verified": True,
        "ensembles": ensembles,
    }


def equal_tensor_archives(first, second, name):
    require(set(first) == set(second), f"{name}: archive keys differ")
    for key in first:
        a, b = first[key], second[key]
        if isinstance(a, dict):
            equal_tensor_archives(a, b, name + "/" + key)
        elif isinstance(a, torch.Tensor):
            require(
                isinstance(b, torch.Tensor) and torch.equal(a, b), f"{name}/{key}: tensors differ"
            )
        else:
            require(a == b, f"{name}/{key}: metadata differs")


def audit_task(task_id, task, saved_scores, saved_decisions, source, run):
    hypothesis, domain, family = task_id.split("/")
    metadata = {"hypothesis": hypothesis, "domain": domain, "family": family}
    require(
        all(task.get(k) == v and saved_scores.get(k) == v for k, v in metadata.items()),
        "Flat task metadata differs",
    )
    require(task.get("status") == "completed", "Task report incomplete")
    folder = run / domain / hypothesis / family
    require(Path(task["run"]).resolve() == folder.resolve(), "Task path differs from domain layout")
    local_report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    require(
        local_report.get("status") == "completed" and local_report.get("task") == family,
        "Local task report differs",
    )
    require(
        local_report["methods"] == task["methods"], "Flat task metrics differ from local report"
    )
    local_scores = torch.load(folder / "scores.pt", weights_only=True)
    equal_tensor_archives(
        local_scores, {k: v for k, v in saved_scores.items() if k not in metadata}, "scores"
    )
    equal_tensor_archives(
        torch.load(folder / "decisions.pt", weights_only=True), saved_decisions, "decisions"
    )
    minutes = vector(saved_scores["minutes"], "minutes")
    labels = vector(saved_scores["labels"], "labels", len(minutes))
    require(
        np.array_equal(labels, source["tasks"][family]["labels"].numpy()),
        "Generated task labels differ",
    )
    require(
        len(minutes) > 1
        and np.all(np.diff(minutes) > 0)
        and np.equal(minutes, minutes.astype(np.int64)).all(),
        "Invalid target chronology",
    )
    require(np.isin(labels, (0, 1)).all(), "Invalid target label")
    cal_minutes = vector(saved_scores["source_calibration_minutes"], "source_calibration_minutes")
    require(
        len(cal_minutes)
        == len(
            source["calibration"] if hypothesis == "h2" else source["role_endpoints"]["calibration"]
        ),
        "Source calibration count differs",
    )
    require(
        len(cal_minutes) >= 19
        and np.all(np.diff(cal_minutes) > 0)
        and cal_minutes.max() < minutes.min(),
        "Invalid/future source calibration",
    )
    methods = expected_methods(hypothesis)
    require(
        set(saved_scores["methods"]) == set(task["methods"]) == methods, "Method coverage differs"
    )
    require(
        set(saved_decisions)
        == {name + "/" + policy for name, policy in product(methods, POLICIES)},
        "Decision policy coverage differs",
    )
    attacks, normals = int((labels == 1).sum()), int((labels == 0).sum())
    if family == "normal_drift":
        require(attacks == 0, "Normal drift includes attacks")
    for name, values in saved_scores["methods"].items():
        energy = vector(values["energy"], name + "/energy", len(minutes))
        cal = vector(values["source_calibration"], name + "/calibration", len(cal_minutes))
        require(set(task["methods"][name]) == set(POLICIES), "Metric policy coverage differs")
        for policy in POLICIES:
            flags, thresholds = independent_replay(energy, cal, minutes, labels, policy)
            stored = saved_decisions[name + "/" + policy]
            require(
                np.array_equal(flags, vector(stored["flags"], name + "/flags", len(minutes))),
                f"{task_id}/{name}/{policy}: flags differ",
            )
            require(
                np.array_equal(
                    thresholds, vector(stored["thresholds"], name + "/thresholds", len(minutes))
                ),
                f"{task_id}/{name}/{policy}: thresholds differ",
            )
            m = task["methods"][name][policy]
            tp, fp = int(flags[labels == 1].sum()), int(flags[labels == 0].sum())
            require(m["tp"] == tp and m["fp"] == fp, "Task counts differ")
            require(
                m["attack"] == m["attack_windows"] == attacks
                and m["normal"] == m["normal_windows"] == normals,
                "Task class coverage differs",
            )
            require(
                m["tpr"] == (tp / attacks if attacks else None)
                and m["fpr"] == (fp / normals if normals else None),
                "Task rates differ",
            )
            require(m["coverage"] == 1.0, "Task coverage incomplete")
            if not attacks or not normals:
                require(m["auroc"] is None and m["pauc_005"] is None, "One-class AUC must be null")
    return {
        **metadata,
        "run": str(folder),
        "methods": task["methods"],
        "target_windows": len(minutes),
        "attack_windows": attacks,
        "normal_windows": normals,
        "policy_conditions_checked": len(methods) * len(POLICIES),
        "all_flags_thresholds_counts_rates_exact": True,
        "negative_control_outcome_used_as_acceptance_gate": False,
        "artifacts": {
            name: {"path": str(folder / name), "sha256": digest(folder / name)}
            for name in ("report.json", "scores.pt", "decisions.pt")
        },
    }


def render_tables(result):
    lines = [
        "# H2·H3 합성 메커니즘 독립 검산",
        "",
        "고차 관계·순서 변경·정상 이동·관측상 동일한 라벨의 진단이다. "
        "새 실제 환경의 일반화나 실장치 양자 우위를 입증하지 않는다.",
        "",
        "각 셀은 TP/FP (TPR% / FPR%)다. 정상 이동은 공격이 없으므로 TPR과 AUC가 null이다. "
        "실제 FPR는 target 라벨로5%에 맞추지 않았다. identical_labels의 성능 차이만으로 "
        "오류를 판정하거나 시드를 버리지 않는다.",
        "",
    ]
    if result["smoke"]:
        lines += ["**축소 통합 스모크이며 전체 정확도 추정이 아니다.**", ""]
    for task_id, task in result["tasks"].items():
        lines += [
            f"## {task_id}",
            "",
            f"공격 {task['attack_windows']}개, 정상 {task['normal_windows']}개.",
            "",
            "| 방법 | source 고정 | 정상10%·5분 지연 | 즉시 정상 확인 |",
            "|---|---:|---:|---:|",
        ]
        for method, policies in task["methods"].items():
            cells = [method]
            for policy in POLICIES:
                m = policies[policy]
                tpr = "null" if m["tpr"] is None else f"{100 * m['tpr']:.2f}"
                fpr = "null" if m["fpr"] is None else f"{100 * m['fpr']:.2f}"
                cells.append(f"{m['tp']}/{m['fp']} ({tpr} / {fpr})")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def run(args):
    folder = Path(args.run)
    require(folder.is_absolute(), "--run must be absolute")
    output = Path(args.output).resolve()
    require(not output.exists(), "Refusing existing audit output")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    registration = json.loads((folder / "registration.json").read_text(encoding="utf-8"))
    require(report.get("status") == "completed", "Mechanism run is incomplete")
    require(
        report.get("smoke") is args.smoke and registration.get("smoke") is args.smoke,
        "Mechanism smoke scope differs",
    )
    require(registration["domains"] == list(DOMAINS.values()), "Registered domain settings differ")
    require(
        registration["seeds"] == ([17] if args.smoke else [17, 29, 41]), "Seed contract differs"
    )
    source_hash = registration.get("source_sha256", "")
    require(
        isinstance(source_hash, str)
        and len(source_hash) == 64
        and all(c in "0123456789abcdef" for c in source_hash),
        "Missing generator source hash",
    )
    require(set(report["domains"]) == set(DOMAINS), "Domain coverage differs")
    scores = torch.load(folder / "scores.pt", weights_only=True)["tasks"]
    decisions = torch.load(folder / "decisions.pt", weights_only=True)["tasks"]
    require(
        set(report["tasks"]) == set(scores) == set(decisions) == expected_tasks(),
        "Exactly all21 registered tasks required",
    )
    torch.set_num_threads(1)
    result = {
        "status": "running",
        "smoke": args.smoke,
        "scope": "synthetic mechanism integration smoke"
        if args.smoke
        else "synthetic mechanism diagnostics",
        "claim": "no independent real-site generalization or quantum-advantage claim",
        "source_run": str(folder),
        "source_generator_sha256": source_hash,
        "audit_source_sha256": digest(__file__),
        "independent_replay_source_sha256": digest(REPLAY_SOURCE),
        "domains": {},
        "tasks": {},
    }
    for domain, config in DOMAINS.items():
        entry = report["domains"][domain]
        require(
            entry["config"] == config
            and entry["observed_input_dimension"] == 2 * config["dimension"],
            "Domain input configuration differs",
        )
        result["domains"][domain] = {"config": config}
        for hypothesis in FAMILIES:
            source, checked = audit_source(
                folder / domain / hypothesis, hypothesis, domain, entry[hypothesis], args.smoke
            )
            require(
                set(entry[hypothesis]["tasks"]) == set(FAMILIES[hypothesis]),
                "Nested domain task coverage differs",
            )
            result["domains"][domain][hypothesis] = checked
            for family in FAMILIES[hypothesis]:
                task_id = f"{hypothesis}/{domain}/{family}"
                task = report["tasks"][task_id]
                nested = entry[hypothesis]["tasks"][family]
                require(
                    nested["methods"] == task["methods"] and nested.get("status") == "completed",
                    "Nested task metrics/status differs",
                )
                result["tasks"][task_id] = audit_task(
                    task_id, task, scores[task_id], decisions[task_id], source, folder
                )
            print(domain, hypothesis, "mechanism audit passed", flush=True)
    total = sum(task["policy_conditions_checked"] for task in result["tasks"].values())
    require(total == 1152, "Incomplete1152-condition mechanism coverage")
    result.update(
        status="completed",
        total_policy_conditions_checked=total,
        artifacts={
            name: {"path": str(folder / name), "sha256": digest(folder / name)}
            for name in ("report.json", "registration.json", "scores.pt", "decisions.pt")
        },
    )
    markdown = render_tables(result)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    with (output / "all_methods.md").open("x", encoding="utf-8") as handle:
        handle.write(markdown)
    print("MECHANISM AUDIT COMPLETE", output / "report.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
