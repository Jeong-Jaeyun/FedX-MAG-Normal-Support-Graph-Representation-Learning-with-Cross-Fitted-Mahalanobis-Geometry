"""Mechanism replay audits permit one-class drift and report null outcomes honestly."""

import importlib.util
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fedxmag_next.h1_experiment import replay

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_h23_mechanisms.py"
SPEC = importlib.util.spec_from_file_location("mechanism_audit", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def matrix(n, dimension):
    return torch.cat((torch.zeros(n, dimension), torch.ones(n, dimension)), 1).double()


def source_draw(hypothesis, dimension):
    counts = audit.expected_counts(hypothesis, True)
    if hypothesis == "h2":
        boundaries = np.cumsum((0, *counts[:4]))
        draws = {role: matrix(counts[i], dimension) for i, role in enumerate(audit.H2_ROLES)}
        for i, role in enumerate(audit.H2_ROLES):
            draws[role][:, :dimension] = (i + 1) / 10
        draws["observation_ids"] = {
            role: torch.arange(boundaries[i], boundaries[i + 1])
            for i, role in enumerate(audit.H2_ROLES)
        }
        draws["role_generator_seeds"] = {
            "train": 1001,
            "selection": 1002,
            "diagnostic_reference": 1003,
            "calibration": 1004,
        }
    else:
        cursor, endpoints = 8, {}
        for role, count in zip(audit.ROLES, counts, strict=True):
            endpoints[role] = torch.arange(cursor, cursor + count)
            cursor += count + 13
        ids = {
            role: torch.unique(query[:, None] - torch.arange(9))
            for role, query in endpoints.items()
        }
        draws = {
            "source": matrix(cursor, dimension),
            "role_endpoints": endpoints,
            "observation_ids": ids,
            "adapter_fit_observations": ids["train"].clone(),
        }
        draws["source"][:, 0] = torch.arange(cursor) / cursor
    draws["tasks"] = {}
    return draws, counts


@pytest.fixture
def mechanism_run(tmp_path):
    run = tmp_path / "mechanisms"
    run.mkdir()
    report = {"status": "completed", "smoke": True, "domains": {}, "tasks": {}}
    flat_scores, flat_decisions = {}, {}
    n = 64
    for domain, config in audit.DOMAINS.items():
        report["domains"][domain] = {
            "config": config,
            "observed_input_dimension": 2 * config["dimension"],
        }
        for hypothesis in audit.FAMILIES:
            folder = run / domain / hypothesis
            folder.mkdir(parents=True)
            draws, counts = source_draw(hypothesis, config["dimension"])
            adapter_fit = (
                draws["train"]
                if hypothesis == "h2"
                else draws["source"][draws["observation_ids"]["train"]]
            )
            adapter = audit.SourcePCA.fit(adapter_fit).state_dict()
            models, states, ensembles = {}, {}, {}
            keys = (
                [
                    f"{corruption}/{kind}/17"
                    for corruption, kind in product(("original", "conditional"), audit.H2_KINDS)
                ]
                if hypothesis == "h2"
                else [f"{kind}/17" for kind in audit.H3_KINDS]
            )
            for key in keys:
                path = folder / (key.replace("/", "_") + ".pt")
                state = (
                    {
                        "seed": 17,
                        "support_fit_role": "train",
                        "support_fit_rows": counts[0],
                        "feature_sd_floor": 1e-6,
                        "geometry": {"counts": torch.tensor([counts[0]])},
                    }
                    if hypothesis == "h2"
                    else {
                        "seed": 17,
                        "model": {
                            "supports": {
                                str(mode): {"geometry": {"counts": torch.tensor([counts[2]])}}
                                for mode in (1, 8)
                            }
                        },
                    }
                )
                state["adapter"] = adapter
                torch.save(state, path)
                states[key] = state
                models[key] = {
                    "path": str(path),
                    "sha256": audit.digest(path),
                    "restore_exact": True,
                }
            if hypothesis == "h2":
                for corruption, kind in product(("original", "conditional"), audit.H2_KINDS):
                    key = f"{corruption}/{kind}"
                    path = folder / f"{corruption}_{kind}_ensemble.pt"
                    torch.save(
                        {
                            "seeds": [states[key + "/17"]],
                            "ensemble_units": {
                                head: {"center": 0.0, "scale": 1.0}
                                for head in ("scalar", "support")
                            },
                        },
                        path,
                    )
                    ensembles[key] = {
                        "path": str(path),
                        "sha256": audit.digest(path),
                        "scaled_errors": {"scalar": 0.0, "support": 0.0},
                    }
            nested_tasks = {}
            for family in audit.FAMILIES[hypothesis]:
                labels = np.zeros(n, int) if family == "normal_drift" else np.tile([0, 1], n // 2)
                # Intentionally perfect null-label separation exercises absence
                # of an outcome-based rejection gate; this is a software fixture.
                energy = (
                    labels.astype(float) if family == "identical_labels" else np.linspace(0, 1, n)
                )
                calibration, minutes = np.zeros(counts[3]), np.arange(100000, 100000 + n)
                method_scores, decisions, method_metrics = {}, {}, {}
                for name in audit.expected_methods(hypothesis):
                    method_scores[name] = {
                        "energy": torch.from_numpy(energy),
                        "source_calibration": torch.from_numpy(calibration),
                    }
                    method_metrics[name] = {}
                    for policy in audit.POLICIES:
                        flags, thresholds = replay(energy, calibration, minutes, labels, policy)
                        attack, normal = labels == 1, labels == 0
                        tp, fp = int(flags[attack].sum()), int(flags[normal].sum())
                        method_metrics[name][policy] = {
                            "tp": tp,
                            "fp": fp,
                            "attack": int(attack.sum()),
                            "normal": int(normal.sum()),
                            "attack_windows": int(attack.sum()),
                            "normal_windows": int(normal.sum()),
                            "tpr": tp / attack.sum() if attack.any() else None,
                            "fpr": fp / normal.sum() if normal.any() else None,
                            "auroc": 0.5 if attack.any() and normal.any() else None,
                            "pauc_005": 0.5 if attack.any() and normal.any() else None,
                            "coverage": 1.0,
                        }
                        decisions[name + "/" + policy] = {
                            "flags": torch.from_numpy(flags),
                            "thresholds": torch.from_numpy(thresholds),
                        }
                task_report = {"status": "completed", "task": family, "methods": method_metrics}
                task_scores = {
                    "methods": method_scores,
                    "minutes": torch.from_numpy(minutes),
                    "labels": torch.from_numpy(labels),
                    "source_calibration_minutes": torch.arange(90000, 90000 + counts[3]),
                }
                task_folder = folder / family
                task_folder.mkdir()
                write_json(task_folder / "report.json", task_report)
                torch.save(task_scores, task_folder / "scores.pt")
                torch.save(decisions, task_folder / "decisions.pt")
                task_id = f"{hypothesis}/{domain}/{family}"
                metadata = {"hypothesis": hypothesis, "domain": domain, "family": family}
                report["tasks"][task_id] = {**metadata, **task_report, "run": str(task_folder)}
                flat_scores[task_id] = {**metadata, **task_scores}
                flat_decisions[task_id] = decisions
                nested_tasks[family] = task_report
                draws["tasks"][family] = {
                    "features": matrix(n + (8 if hypothesis == "h3" else 0), config["dimension"]),
                    "labels": torch.from_numpy(labels),
                }
            torch.save(draws, folder / "draws.pt")
            report["domains"][domain][hypothesis] = {
                "models": models,
                "ensemble_audits": ensembles,
                "tasks": nested_tasks,
                "counts": counts,
                "observation_roles_disjoint": True,
            }
    write_json(run / "report.json", report)
    write_json(
        run / "registration.json",
        {
            "smoke": True,
            "domains": list(audit.DOMAINS.values()),
            "seeds": [17],
            "source_sha256": "1" * 64,
        },
    )
    torch.save({"tasks": flat_scores}, run / "scores.pt")
    torch.save({"tasks": flat_decisions}, run / "decisions.pt")
    return run


def test_literal_task_method_and_budget_contract():
    assert len(audit.expected_tasks()) == 21
    assert len(audit.expected_methods("h2")) == 20
    assert len(audit.expected_methods("h3")) == 16
    assert (
        sum(len(audit.expected_methods(task.split("/")[0])) * 3 for task in audit.expected_tasks())
        == 1152
    )
    assert audit.expected_counts("h2", False) == (4096, 1024, 1024, 256, 1024)
    assert audit.expected_counts("h3", False) == (256, 128, 128, 128)


def test_all1152_decisions_and_one_class_drift_pass(mechanism_run, tmp_path):
    output = tmp_path / "audit"
    audit.run(SimpleNamespace(run=str(mechanism_run), output=str(output), smoke=True))
    result = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed" and result["total_policy_conditions_checked"] == 1152
    assert len(result["tasks"]) == 21
    drift = result["tasks"]["h2/d8_m1/normal_drift"]
    assert drift["attack_windows"] == 0
    assert all(
        m["fixed"]["tpr"] is None and m["fixed"]["auroc"] is None for m in drift["methods"].values()
    )
    null = result["tasks"]["h2/d8_m1/identical_labels"]
    assert null["negative_control_outcome_used_as_acceptance_gate"] is False
    assert all(m["fixed"]["tpr"] == 1 and m["fixed"]["fpr"] == 0 for m in null["methods"].values())
    for domain in audit.DOMAINS:
        for hypothesis in audit.FAMILIES:
            checked = result["domains"][domain][hypothesis]
            actual_draws = mechanism_run / domain / hypothesis / "draws.pt"
            assert Path(checked["draws"]["path"]) == actual_draws
            assert checked["draws"]["sha256"] == audit.digest(actual_draws)
            assert checked["adapter_train_only_verified"] is True
            pca = checked["adapter_train_audit"]
            assert pca["adapters_checked"] == (20 if hypothesis == "h2" else 8)
            assert max(pca["max_absolute_errors"].values()) == 0
            if hypothesis == "h2":
                assert checked["history_observation_membership_exact"] is None
            else:
                assert checked["history_observation_membership_exact"] is True
    markdown = (output / "all_methods.md").read_text(encoding="utf-8")
    assert "null" in markdown and "스모크" in markdown


@pytest.mark.parametrize(
    "mutation",
    [
        "flags",
        "model_hash",
        "support_budget",
        "role_overlap",
        "history_leak",
        "selection_pca",
        "h3_selection_pca",
        "missing_task",
        "smoke_scope",
    ],
)
def test_rejects_incomplete_or_changed_evidence(mechanism_run, tmp_path, mutation):
    report_path = mechanism_run / "report.json"
    report = json.loads(report_path.read_text())
    if mutation == "flags":
        path = mechanism_run / "decisions.pt"
        saved = torch.load(path, weights_only=True)
        first = next(iter(saved["tasks"]["h2/d8_m1/parity_flip"].values()))
        first["flags"][0] = ~first["flags"][0]
        torch.save(saved, path)
    elif mutation == "model_hash":
        entry = next(iter(report["domains"]["d8_m1"]["h2"]["models"].values()))
        with Path(entry["path"]).open("ab") as handle:
            handle.write(b"changed")
    elif mutation == "support_budget":
        entry = next(iter(report["domains"]["d8_m1"]["h2"]["models"].values()))
        state = torch.load(entry["path"], weights_only=True)
        state["geometry"]["counts"] = torch.tensor([32])
        torch.save(state, entry["path"])
        entry["sha256"] = audit.digest(entry["path"])
        write_json(report_path, report)
    elif mutation in ("role_overlap", "history_leak"):
        hypothesis = "h2" if mutation == "role_overlap" else "h3"
        path = mechanism_run / "d8_m1" / hypothesis / "draws.pt"
        saved = torch.load(path, weights_only=True)
        if mutation == "role_overlap":
            saved["observation_ids"]["selection"][0] = saved["observation_ids"]["train"][0]
        else:
            saved["adapter_fit_observations"] = torch.cat(
                (saved["adapter_fit_observations"], saved["observation_ids"]["selection"][:1])
            )
        torch.save(saved, path)
    elif mutation in ("selection_pca", "h3_selection_pca"):
        hypothesis = "h2" if mutation == "selection_pca" else "h3"
        entry = next(iter(report["domains"]["d8_m1"][hypothesis]["models"].values()))
        draws = torch.load(mechanism_run / "d8_m1" / hypothesis / "draws.pt", weights_only=True)
        selection = (
            draws["selection"]
            if hypothesis == "h2"
            else draws["source"][draws["observation_ids"]["selection"]]
        )
        state = torch.load(entry["path"], weights_only=True)
        state["adapter"] = audit.SourcePCA.fit(selection).state_dict()
        torch.save(state, entry["path"])
        entry["sha256"] = audit.digest(entry["path"])
        write_json(report_path, report)
    elif mutation == "missing_task":
        report["tasks"].pop("h2/d8_m1/parity_flip")
        write_json(report_path, report)
    else:
        report["smoke"] = False
        write_json(report_path, report)
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="PCA" if "selection_pca" in mutation else None):
        audit.run(SimpleNamespace(run=str(mechanism_run), output=str(output), smoke=True))
    assert not output.exists()


def test_each_pca_field_is_checked_against_actual_train_statistics():
    train = torch.randn(64, 16, generator=torch.Generator().manual_seed(21)).double()
    reference = audit.SourcePCA.fit(train)
    state = reference.state_dict()
    for field in state:
        changed = {name: value.clone() for name, value in state.items()}
        changed[field][(0,) * changed[field].ndim] += 1e-4
        with pytest.raises(ValueError, match=f"PCA {field}"):
            audit.verify_adapter(reference, changed, "tampered")
