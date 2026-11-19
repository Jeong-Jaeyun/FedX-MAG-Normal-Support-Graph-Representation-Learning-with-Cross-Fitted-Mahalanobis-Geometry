import json
from pathlib import Path

import pytest
import torch
import yaml

from fedxmag.config import config_fingerprint, load_stage1_config
from fedxmag.confirmation import model_builders
from fedxmag.m1.experiment import geometry, load_context, register, sealed_test, train_arm
from fedxmag.schema import RESOLUTION_AWARE_FEATURE_GROUPS
from fedxmag.serialization import save_graph_samples
from fedxmag.synthetic import make_synthetic_graphs

ROOT = Path(__file__).resolve().parents[1]


def _project(tmp_path: Path, decision_status: str = "pass") -> Path:
    groups = RESOLUTION_AWARE_FEATURE_GROUPS
    config = yaml.safe_load((ROOT / "configs" / "stage1_m0_v3_development.yaml").read_text())
    config["model"].update({"hidden_dim": 8, "graph_dim": 4, "num_layers": 1, "dropout": 0.0})
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text(yaml.safe_dump(config))
    fingerprint = config_fingerprint(load_stage1_config(tmp_path / "configs" / "base.yaml"))

    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for index, (name, count) in enumerate(
        (("train", 20), ("select", 6), ("calibration", 10), ("test_normal", 8), ("test_attack", 8))
    ):
        samples = make_synthetic_graphs(num_graphs=count, seed=100 + index)
        for position, sample in enumerate(samples):
            sample.metadata.update(
                {"session_key": f"date-{position % 3}", "event_count": 10 + position}
            )
            if name == "test_attack":
                sample.label = 1
                sample.node_features[:, :] += 3.0
                sample.metadata["attack_categories"] = ["Recon" if position % 2 else "DoS"]
        save_graph_samples(prepared / f"{name}.pt", samples, groups=groups)

    m0_dir = tmp_path / "m0"
    m0_dir.mkdir()
    builders = model_builders(config, groups)
    for predictor in ("graph", "set_context", "shuffled_graph"):
        for seed in (1, 2):
            torch.manual_seed(seed)
            torch.save({"model": builders[predictor]().state_dict()},
                       m0_dir / f"m0_{predictor}_seed_{seed}.pt")
    (m0_dir / "confirmation_decision.json").write_text(json.dumps({"status": decision_status}))

    protocol = yaml.safe_load((ROOT / "configs" / "m1_protocol.yaml").read_text())
    protocol["protocol"]["prerequisite_m0_decision"] = "m0/confirmation_decision.json"
    protocol["base_config"] = "configs/base.yaml"
    protocol["base_config_sha256"] = fingerprint
    protocol["m0_confirmation_dir"] = "m0"
    protocol["m0_confirmation_seeds"] = [1, 2]
    protocol["data"]["prepared_dir"] = "prepared"
    protocol["data"]["roles"] = {
        "fit": "train.pt",
        "selection": "select.pt",
        "calibration": "calibration.pt",
        "test_normal": "test_normal.pt",
        "test_attack": "test_attack.pt",
    }
    protocol["data"]["file_sha256"] = {}
    protocol["training"].update(
        {"seeds": [1, 2], "batch_size": 8, "validation_batch_size": 4, "epochs": 2}
    )
    protocol["objective_defaults"]["rank_min"] = 2.0
    protocol["arms"] = {
        name: protocol["arms"][name] for name in ("m0_recon", "m1_lw", "ns_only_lw")
    }
    protocol["test"]["primary_comparisons"] = [["m1_lw", "m0_recon"]]
    protocol["test"]["bootstrap_resamples"] = 20
    protocol["geometry"]["split_stability_resamples"] = 2
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol))
    return path


def test_m1_phases_run_end_to_end_and_refuse_reruns(tmp_path: Path) -> None:
    protocol = _project(tmp_path)
    context = load_context(protocol, tmp_path, tmp_path / "out")
    register(context)
    for arm in context.arms:
        for seed in context.seeds:
            summary = train_arm(context, arm, seed)
            assert summary["epochs_run"] == 2
    with pytest.raises(FileExistsError):
        train_arm(context, "m1_lw", 1)
    report = geometry(context)
    member = report["methods"]["m1_lw"]["members"]["1"]
    assert set(member["calibration_thresholds"]) == {"0.01", "0.05", "0.1"}
    assert "training_alpha_trajectory_mean" in member["shrinkage_replay"]
    assert "seed_stability" in report["methods"]["m1_lw"]
    assert set(report["representation_change_cka_vs_m0_recon"]["m1_lw"]) == {"1", "2"}
    result = sealed_test(context)
    assert result["counts"] == {"test_normal": 8, "test_attack": 8, "calibration": 10}
    lw = result["methods"]["m1_lw"]
    assert 0.0 <= lw["aggregate_over_members"]["auroc"]["mean"] <= 1.0
    assert set(lw["members"]["1"]["per_category_tpr"]) == {"DoS", "Recon"}
    assert "m1_lw__minus__m0_recon" in result["paired_cluster_bootstrap"]["differences"]
    assert "handcrafted_combined_lw" in result["methods"]
    with pytest.raises(FileExistsError):
        sealed_test(context)


def test_m1_registration_is_blocked_unless_m0_passed(tmp_path: Path) -> None:
    protocol = _project(tmp_path, decision_status="fail")
    context = load_context(protocol, tmp_path, tmp_path / "out")
    with pytest.raises(RuntimeError, match="blocked"):
        register(context)


def test_real_m1_protocol_parses_with_every_arm() -> None:
    context = load_context(ROOT / "configs" / "m1_protocol.yaml", ROOT, ROOT / "unused")
    assert context.protocol["protocol"]["primary_arm"] == "m1_lw"
    assert context.settings["m0_recon"].uses_normal_support is False
    assert context.settings["m1_lw_no_cov_reg"].lambda_c == 0.0
    assert context.settings["m1_lw_in_sample"].energy_mode == "in_sample"
    assert context.settings["ns_only_lw"].reconstruction_weight == 0.0
    assert context.settings["m1_empirical"].estimator == "empirical"
