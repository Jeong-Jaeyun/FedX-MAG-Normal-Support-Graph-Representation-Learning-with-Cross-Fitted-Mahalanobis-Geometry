"""Independent replay must reject mutated artifacts and preserve causality."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_h23_results.py"
SPEC = importlib.util.spec_from_file_location("audit_h23_results", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def make_run(path):
    path.mkdir()
    minutes = np.arange(100, 106)
    labels = np.array([0, 1, 0, 1, 0, 1])
    source = np.arange(50, dtype=float)
    energy = np.array([49.0, 75.0, 10.0, 35.0, 90.0, 50.0])
    methods, decisions = {}, {}
    for policy in audit.POLICIES:
        flags, thresholds = audit.independent_replay(energy, source, minutes, labels, policy)
        tp, fp = int(flags[labels == 1].sum()), int(flags[labels == 0].sum())
        methods[policy] = {
            "tp": tp,
            "fp": fp,
            "attack_windows": 3,
            "normal_windows": 3,
            "coverage": 1.0,
            "tpr": tp / 3,
            "fpr": fp / 3,
        }
        for name in audit.registered_methods("h2", smoke=True):
            decisions[name + "/" + policy] = {
                "flags": torch.from_numpy(flags.copy()),
                "thresholds": torch.from_numpy(thresholds.copy()),
            }
    (path / "report.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "dataset": "cic",
                "smoke": True,
                "methods": {name: methods for name in audit.registered_methods("h2", smoke=True)},
                "audits": {},
                "training": {},
            }
        )
    )
    (path / "registration.json").write_text('{"smoke_only":true}')
    torch.save(
        {
            "minutes": torch.from_numpy(minutes),
            "labels": torch.from_numpy(labels),
            "types": [[], ["attack"], [], ["attack"], [], ["attack"]],
            "source_calibration_minutes": torch.arange(50),
            "methods": {
                name: {
                    "energy": torch.from_numpy(energy),
                    "source_calibration": torch.from_numpy(source),
                }
                for name in audit.registered_methods("h2", smoke=True)
            },
        },
        path / "scores.pt",
    )
    torch.save(decisions, path / "decisions.pt")
    torch.save(
        {
            "source_role_indices": {
                "train": torch.arange(10),
                "selection": torch.arange(10, 20),
                "calibration": torch.arange(20, 30),
            }
        },
        path / "source_adapter.pt",
    )
    return {"hypothesis": "h2", "dataset": "cic", "run": str(path)}


def test_current_label_does_not_change_own_threshold_or_decision():
    source = np.arange(50, dtype=float)
    minutes, labels = np.arange(200, 240), np.zeros(40, dtype=int)
    energy = np.linspace(0, 100, 40)
    for policy in audit.POLICIES:
        flags, thresholds = audit.independent_replay(energy, source, minutes, labels, policy)
        for index in range(len(labels)):
            changed = labels.copy()
            changed[index] = 1
            other_flags, other_thresholds = audit.independent_replay(
                energy, source, minutes, changed, policy
            )
            assert np.array_equal(flags[: index + 1], other_flags[: index + 1])
            assert np.array_equal(thresholds[: index + 1], other_thresholds[: index + 1])


def test_known_finite_sample_rank_and_strict_comparison():
    flags, thresholds = audit.independent_replay(
        np.array([47.0, 48.0]), np.arange(50), np.array([100, 101]), np.array([0, 1]), "fixed"
    )
    # ceil(51*.95)=49, so zero-based index48 has value48; equality is not alarm.
    assert np.array_equal(thresholds, [48.0, 48.0])
    assert not flags.any()


def test_actual_corrupted_decision_rejected(tmp_path):
    spec = make_run(tmp_path / "run")
    result = audit.audit_run(spec, tmp_path, smoke=True)
    assert result["all_decisions_thresholds_counts_rates_exact"] is True
    assert result["policy_conditions_checked"] == 123
    path = Path(spec["run"]) / "decisions.pt"
    decisions = torch.load(path, weights_only=True)
    decisions["raw_m8_reference/immediate"]["flags"][2] ^= True
    torch.save(decisions, path)
    with pytest.raises(ValueError, match="flags differ"):
        audit.audit_run(spec, tmp_path, smoke=True)


def test_smoke_cannot_be_read_as_production(tmp_path):
    spec = make_run(tmp_path / "run")
    with pytest.raises(ValueError, match="smoke scope mismatch"):
        audit.audit_run(spec, tmp_path, smoke=False)


def test_future_calibration_or_missing_method_rejected(tmp_path):
    spec = make_run(tmp_path / "run")
    path = Path(spec["run"]) / "scores.pt"
    scores = torch.load(path, weights_only=True)
    scores["source_calibration_minutes"][-1] = 101
    torch.save(scores, path)
    with pytest.raises(ValueError, match="Future calibration"):
        audit.audit_run(spec, tmp_path, smoke=True)


def test_overlapping_source_ids_rejected(tmp_path):
    spec = make_run(tmp_path / "run")
    path = Path(spec["run"]) / "source_adapter.pt"
    saved = torch.load(path, weights_only=True)
    saved["source_role_indices"]["selection"][0] = 0
    torch.save(saved, path)
    with pytest.raises(ValueError, match="Overlapping source"):
        audit.audit_run(spec, tmp_path, smoke=True)


def test_missing_whole_preregistered_method_rejected_even_if_archives_agree(tmp_path):
    spec = make_run(tmp_path / "run")
    folder = Path(spec["run"])
    missing = "original_quantum_scalar"
    report = json.loads((folder / "report.json").read_text())
    del report["methods"][missing]
    (folder / "report.json").write_text(json.dumps(report))
    scores = torch.load(folder / "scores.pt", weights_only=True)
    del scores["methods"][missing]
    torch.save(scores, folder / "scores.pt")
    decisions = torch.load(folder / "decisions.pt", weights_only=True)
    for policy in audit.POLICIES:
        del decisions[missing + "/" + policy]
    torch.save(decisions, folder / "decisions.pt")
    with pytest.raises(ValueError, match="Preregistered method coverage"):
        audit.audit_run(spec, tmp_path, smoke=True)


def test_expected_hypothesis_method_counts_are_fixed():
    assert len(audit.registered_methods("h2", smoke=True)) == 41
    assert len(audit.registered_methods("h2", smoke=False)) == 42
    assert len(audit.registered_methods("h3")) == 16
    with pytest.raises(ValueError, match="Unknown registered"):
        audit.registered_methods("h4")
