"""Post-run evidence remains paired, complete and distinct from smoke results."""

import importlib.util
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_h23_findings.py"
SPEC = importlib.util.spec_from_file_location("h23_findings", SCRIPT)
findings = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(findings)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_pairing_counts_changes_instead_of_only_net_accuracy():
    labels = np.array([1, 1, 1, 0, 0, 0])
    actual = findings.paired_difference([1, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 0], labels)
    assert [actual[k] for k in ("added_tp", "lost_tp", "added_fp", "removed_fp")] == [1, 1, 1, 1]
    assert actual["delta_tpr"] == actual["delta_fpr"] == 0
    assert not actual["candidate_fpr_within_5pct"]
    with pytest.raises(ValueError, match="shapes"):
        findings.paired_difference([1], [1, 0], [1, 0])


def test_practical_gate_checks_false_alarms_all_three_and_fixed_boundaries():
    entries = {
        name: {"delta_tpr": change, "candidate_fpr": 0.05}
        for name, change in zip(findings.DATASETS, [0.02, 0.02, -0.01], strict=True)
    }
    assert findings.practical_screen(entries)["passed"]
    entries["ton"]["candidate_fpr"] = 0.0501
    assert not findings.practical_screen(entries)["passed"]
    entries["ton"]["candidate_fpr"] = 0.04
    entries["ton"]["delta_tpr"] = -0.0101
    assert not findings.practical_screen(entries)["passed"]
    with pytest.raises(ValueError, match="three"):
        findings.practical_screen({"cic": entries["cic"]})


def test_fixed_comparisons_keep_unequal_input_and_head_effects_separate():
    h2 = findings.comparison_specs("h2")
    assert ("original_quantum_scalar", "original_quantum_support", "head", False) in h2
    assert ("conditional_mlp32_scalar", "original_mlp32_scalar", "corruption", False) in h2
    assert (
        "original_quantum_scalar",
        "frozen_dnn20k_m8_reference",
        "unequal_20k_reference",
        False,
    ) in h2
    gated = [entry for entry in h2 if entry[3]]
    assert len(gated) == 12
    assert {entry[2] for entry in gated} == {"same_input"}
    assert all("full_mlp64" not in entry[1] for entry in gated)
    h3 = findings.comparison_specs("h3")
    assert len(h3) == 14
    assert len([entry for entry in h3 if entry[3]]) == 7
    assert ("quantum_m8", "quantum_zero_history_m8", "temporal_primary", True) in h3
    assert ("quantum_m1", "quantum_shuffled_m1", "mode1_sensitivity", False) in h3


def test_selected_epoch_zero_is_preserved_and_missing_checkpoint_rejected():
    trained = {
        "selected_epoch": 0,
        "history": [
            {"epoch": 0, "selection_synthetic_bce": 0.69},
            {"epoch": 1, "selection_synthetic_bce": 0.70},
        ],
        "parameter_count": 65,
        "parameter_l2_change": 0.0,
        "train_corruption_audit": {"changed_row_fraction": 0.8},
        "validation_corruption_audit": {"changed_row_fraction": 0.7},
    }
    result = findings.training_summary({"original/quantum/17": trained})["original/quantum/17"]
    assert result["selected_epoch"] == 0 and result["epochs_run"] == 1
    assert result["selection_bce_change_from_initial"] == 0
    trained["selected_epoch"] = 5
    with pytest.raises(ValueError, match="checkpoint"):
        findings.training_summary({"original/quantum/17": trained})


def method_names(hypothesis):
    if hypothesis == "h2":
        names = {"raw_m8_reference", "frozen_dnn20k_m8_reference"}
        for corruption, kind, head in product(
            ("original", "conditional"), findings.KINDS, ("scalar", "support")
        ):
            branch = f"{corruption}_{kind}_{head}"
            names.update((branch, "raw_plus_" + branch))
        return sorted(names)
    return [f"{kind}_m{mode}" for kind, mode in product(findings.H3_KINDS, (1, 8))]


@pytest.fixture
def full_audit(tmp_path):
    entries = []
    post = {
        "normal_with_attack_in_previous8min": {"windows": 1, "false_alarms": 1, "fpr": 1.0},
        "other_normal": {"windows": 1, "false_alarms": 0, "fpr": 0.0},
        "contiguous_attack_to_normal_runs": [],
    }
    for hypothesis, dataset in product(("h2", "h3"), findings.DATASETS):
        folder = tmp_path / f"{hypothesis}_{dataset}"
        folder.mkdir()
        metrics = {
            "tp": 1,
            "fp": 1,
            "attack_windows": 2,
            "normal_windows": 2,
            "tpr": 0.5,
            "fpr": 0.5,
            "coverage": 1.0,
            "per_type": {"example": {"windows": 2, "detected": 1, "tpr": 0.5}},
            "post_attack": post,
        }
        methods = {
            name: {policy: metrics for policy in findings.POLICIES}
            for name in method_names(hypothesis)
        }
        report = {
            "status": "completed",
            "dataset": dataset,
            "methods": methods,
            "audits": {"quantum": {"prediction_fit": [{"validation_mse": 0.12}]}},
            "training": {},
            "elapsed_seconds": 1.0,
            "source_counts": {
                "train": 100,
                "selection": 40,
                "residual_support": 40,
                "calibration": 40,
            },
            "target_cold_start_windows": 1,
            "target_missing_current": 0,
        }
        registration = {
            "fit_rows": 4096,
            "validation_rows": 4096,
            "selection_rows": 5000,
            "input_audit": {"example_counts": report["source_counts"]},
        }
        for name, value in (("report.json", report), ("registration.json", registration)):
            (folder / name).write_text(json.dumps(value), encoding="utf-8")
        torch.save({"labels": torch.tensor([0, 1, 0, 1])}, folder / "scores.pt")
        torch.save(
            {
                name + "/" + policy: {"flags": torch.tensor([1, 1, 0, 0], dtype=torch.bool)}
                for name, policy in product(methods, findings.POLICIES)
            },
            folder / "decisions.pt",
        )
        torch.save(
            {"adapter": {"explained_variance_ratio": torch.ones(8).double() / 10}},
            folder / "source_adapter.pt",
        )
        entries.append(
            {
                "hypothesis": hypothesis,
                "dataset": dataset,
                "run": str(folder),
                "methods": methods,
                "target_windows": 4,
                "attack_windows": 2,
                "normal_windows": 2,
                "all_decisions_thresholds_counts_rates_exact": True,
                "all_source_observation_roles_disjoint": True,
                "full_target_census_matches_h1": True,
                "artifacts": {
                    name: {"path": str(folder / name), "sha256": findings.digest(folder / name)}
                    for name in (
                        "report.json",
                        "registration.json",
                        "scores.pt",
                        "decisions.pt",
                        "source_adapter.pt",
                    )
                },
                "paired_quantum_noent": {"saved_interval": "preserved"},
            }
        )
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "smoke": False,
                "scope": "full_prepared_target_census",
                "runs": entries,
                "total_policy_conditions_checked": 522,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_full_verified_summary_preserves_all_methods_and_diagnostics(full_audit, tmp_path):
    output = tmp_path / "findings"
    findings.run(SimpleNamespace(audit=str(full_audit), output=str(output)))
    result = json.loads((output / "findings.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed" and result["policy_conditions"] == 522
    assert len(result["runs"]) == 6 and len(result["practical_screens"]) == 57
    assert [len(run["methods"]) for run in result["runs"]] == [42] * 3 + [16] * 3
    assert all(
        run["audited_quantum_noent_bootstrap"] == {"saved_interval": "preserved"}
        for run in result["runs"]
    )
    temporal = result["runs"][3]
    assert (
        temporal["methods"]["quantum_m8"]["immediate"]["post_attack"]["other_normal"]["windows"]
        == 1
    )
    assert temporal["source_context"]["prediction_fit"]["quantum"][0]["validation_mse"] == 0.12
    assert result["audit"]["sha256"] == findings.digest(full_audit)
    markdown = (output / "findings.md").read_text(encoding="utf-8")
    assert "⚠" in markdown and "1모드는 민감도" in markdown
    assert all(not screen["passed"] for screen in result["practical_screens"].values())
    with pytest.raises(ValueError, match="overwrite"):
        findings.run(SimpleNamespace(audit=str(full_audit), output=str(output)))


@pytest.mark.parametrize("mutation", ["smoke", "incomplete", "missing_run", "changed_artifact"])
def test_rejects_unverified_or_changed_evidence_without_output(full_audit, tmp_path, mutation):
    value = json.loads(full_audit.read_text())
    if mutation == "smoke":
        value["smoke"] = True
    elif mutation == "incomplete":
        value["status"] = "running"
    elif mutation == "missing_run":
        value["runs"].pop()
    else:
        Path(value["runs"][0]["artifacts"]["report.json"]["path"]).write_text("changed")
    full_audit.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "rejected_output"
    with pytest.raises(ValueError):
        findings.run(SimpleNamespace(audit=str(full_audit), output=str(output)))
    assert not output.exists()
