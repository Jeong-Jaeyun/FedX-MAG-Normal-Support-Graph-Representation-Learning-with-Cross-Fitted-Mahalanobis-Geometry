import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from fedxmag.confirmation import (
    CONTROLS,
    PREDICTORS,
    cluster_bootstrap,
    evaluate_confirmation,
    family_gate_decision,
    load_protocol,
    model_builders,
    train_predictors_for_seed,
)
from fedxmag.losses import FAMILY_KEYS
from fedxmag.masking import FamilyMasker
from fedxmag.schema import DEFAULT_FEATURE_GROUPS
from fedxmag.synthetic import make_synthetic_graphs

ROOT = Path(__file__).resolve().parents[1]
GATES = {
    "minimum_graph_gain_over_local": 0.05,
    "minimum_graph_gain_over_set_context": 0.05,
    "minimum_graph_gain_over_shuffled_graph": 0.05,
    "require_graph_better_than_constant": True,
    "maximum_standardized_graph_mse": 1.0,
}
TINY_CONFIG = {
    "model": {"hidden_dim": 8, "graph_dim": 4, "num_layers": 1, "dropout": 0.0},
    "controls": {"adjacency_shuffle_seed": 5, "endpoint_shuffle_mode": "both_endpoints"},
    "training": {
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "gradient_clip_norm": 5.0,
        "early_stopping_patience": 5,
        "validation_mask_seeds": [3, 5],
        "selection_metric": "validation_mask_mean_worst_family_mse",
    },
    "loss": {key: 1.0 for key in FAMILY_KEYS},
}


def _samples(count: int, seed: int) -> list:
    samples = make_synthetic_graphs(num_graphs=count, seed=seed)
    for index, sample in enumerate(samples):
        sample.metadata["session_key"] = f"session-{index % 3}"
    return samples


def test_gate_function_reproduces_archived_v2_audit_decision() -> None:
    path = ROOT / "output" / "m0" / "xiiotid_stage1_v2" / "pretext_audit.json"
    if not path.exists():
        pytest.skip("archived v2 audit report is not available")
    report = json.loads(path.read_text(encoding="utf-8"))
    per_seed = {
        predictor: [run[predictor]["aggregate"] for run in report["per_model_seed"]]
        for predictor in PREDICTORS
    }
    decision = family_gate_decision(report["constant"]["aggregate"], per_seed, GATES)
    assert decision["status"] == report["status"] == "fail"
    for key in FAMILY_KEYS:
        archived = report["criteria"]["families"][key]
        ours = decision["families"][key]
        assert ours["pass"] == archived["pass"]
        for control in CONTROLS:
            assert math.isclose(
                ours[f"graph_gain_over_{control}"],
                archived[f"graph_gain_over_{control}"],
                rel_tol=1e-12,
            )


def test_gate_requires_every_family_and_every_control() -> None:
    def aggregate(value: float) -> dict[str, float]:
        return {f"{key}_mse_mean": value for key in FAMILY_KEYS}

    per_seed = {
        "local": [aggregate(1.0)] * 2,
        "set_context": [aggregate(0.80)] * 2,
        "shuffled_graph": [aggregate(0.80)] * 2,
        "graph": [aggregate(0.70)] * 2,
    }
    assert family_gate_decision(aggregate(1.2), per_seed, GATES)["status"] == "pass"
    per_seed["shuffled_graph"] = [aggregate(0.72)] * 2
    failed = family_gate_decision(aggregate(1.2), per_seed, GATES)
    assert failed["status"] == "fail"
    assert all(not entry["shuffled_graph_gain_pass"] for entry in failed["families"].values())
    per_seed["shuffled_graph"] = [aggregate(0.80)] * 2
    per_seed["graph"] = [aggregate(1.05)] * 2
    per_seed["local"] = per_seed["set_context"] = [aggregate(2.0)] * 2
    per_seed["shuffled_graph"] = [aggregate(2.0)] * 2
    too_high = family_gate_decision(aggregate(3.0), per_seed, GATES)
    assert too_high["status"] == "fail"
    assert not too_high["families"]["node_timing"]["predictability_pass"]


def test_cluster_bootstrap_recovers_exact_gain_and_handles_missing_targets() -> None:
    rng = np.random.default_rng(0)
    control = rng.uniform(0.5, 1.5, size=(2, 3, 12, 4))
    control[0, 0, 0, 1] = np.nan
    errors = {name: control.copy() for name in CONTROLS}
    errors["graph"] = 0.5 * control
    summary = cluster_bootstrap(
        errors,
        control[0],
        [f"c{index % 4}" for index in range(12)],
        resamples=64,
        seed=1,
        interval=0.95,
    )
    for key in FAMILY_KEYS:
        for name in CONTROLS:
            interval = summary[key][f"graph_gain_over_{name}"]
            assert interval["lower"] == pytest.approx(0.5)
            assert interval["upper"] == pytest.approx(0.5)
            assert interval["probability_gain_below_5_percent"] == 0.0


def test_training_and_evaluation_core_reproduce_frozen_aggregates() -> None:
    groups = DEFAULT_FEATURE_GROUPS
    train_samples = _samples(8, 51)
    select_samples = _samples(4, 52)
    confirmation_samples = _samples(9, 53)
    masker = FamilyMasker(0.35, 0.8, groups=groups)
    builders = model_builders(TINY_CONFIG, groups)
    seeds = [7, 11]
    models = {}
    for seed in seeds:
        trained, runs = train_predictors_for_seed(
            builders, train_samples, select_samples, masker, TINY_CONFIG, seed, groups
        )
        assert set(trained) == set(PREDICTORS)
        assert all(run["epochs_run"] == 1 for run in runs.values())
        models.update({(name, seed): model for name, model in trained.items()})
    decisions = []
    result = evaluate_confirmation(
        models,
        confirmation_samples,
        masker,
        seeds,
        [101, 103, 107],
        GATES,
        {"bootstrap_resamples": 32, "bootstrap_seed": 3, "interval": 0.9},
        batch_size=4,
        groups=groups,
        on_decision=decisions.append,
    )
    assert decisions == [result["decision"]]
    secondary = result["secondary_descriptive"]
    assert secondary["per_graph_reproduction_max_relative_discrepancy"] <= 1e-5
    assert result["errors"]["graph"].shape == (2, 3, 9, 4)
    assert result["constant_errors"].shape == (3, 9, 4)
    for key in FAMILY_KEYS:
        for control in CONTROLS:
            assert secondary["per_mask_wins"][key][control]["comparisons"] == 6
    assert set(secondary["leave_one_seed_out"]) == {"without_7", "without_11"}
    assert secondary["session_cluster_bootstrap"]["clusters"] == 3


def test_trained_models_are_deterministic_for_a_seed() -> None:
    groups = DEFAULT_FEATURE_GROUPS
    train_samples = _samples(6, 61)
    select_samples = _samples(3, 62)
    masker = FamilyMasker(0.35, 0.8, groups=groups)
    builders = {"graph": model_builders(TINY_CONFIG, groups)["graph"]}
    first, _ = train_predictors_for_seed(
        builders, train_samples, select_samples, masker, TINY_CONFIG, 5, groups
    )
    second, _ = train_predictors_for_seed(
        builders, train_samples, select_samples, masker, TINY_CONFIG, 5, groups
    )
    for left, right in zip(
        first["graph"].state_dict().values(), second["graph"].state_dict().values(), strict=True
    ):
        assert torch.equal(left, right)


def test_protocol_rejects_a_changed_gate(tmp_path: Path) -> None:
    protocol_path = ROOT / "configs" / "m0_v3b_confirmation_protocol.yaml"
    prepared = ROOT / "data" / "processed" / "xiiotid_stage1_v3b_development"
    if not prepared.exists():
        pytest.skip("prepared v3b artifact is not available")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol["gates"]["minimum_graph_gain_over_shuffled_graph"] = 0.01
    changed = tmp_path / "protocol.yaml"
    changed.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="gate"):
        load_protocol(changed, ROOT)


def test_protocol_rejects_a_forbidden_confirmation_role(tmp_path: Path) -> None:
    protocol_path = ROOT / "configs" / "m0_v3b_confirmation_protocol.yaml"
    prepared = ROOT / "data" / "processed" / "xiiotid_stage1_v3b_development"
    if not prepared.exists():
        pytest.skip("prepared v3b artifact is not available")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    protocol["data"]["roles"]["confirmation"] = "audit_normal.pt"
    changed = tmp_path / "protocol.yaml"
    changed.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        load_protocol(changed, ROOT)
