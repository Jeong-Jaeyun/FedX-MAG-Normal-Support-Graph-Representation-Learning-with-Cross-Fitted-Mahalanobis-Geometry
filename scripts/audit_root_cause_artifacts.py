"""Check frozen inputs, score replay, split isolation, and bind research artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import torch
from investigate_context import ROLES
from investigate_detection import DEFAULT_OUTPUT, PREPARED, ROLE_PATHS, ROOT, write_json
from run_entity_detector import verified_inputs

from fedxmag.config import (
    config_fingerprint,
    load_stage1_config,
    preprocessing_code_fingerprint,
    training_code_fingerprint,
)
from fedxmag.confirmation import sha256_file
from fedxmag.scoring import FrozenNodeDetector
from fedxmag.serialization import load_graph_samples


def main():
    output = DEFAULT_OUTPUT
    destination = output / "validation_manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    protocol, _ = verified_inputs()
    registration = json.loads(
        (ROOT / "output/m0/xiiotid_stage1_v3b_confirmation/registration.json").read_text()
    )
    frozen = registration["frozen_inputs"]
    assert (
        config_fingerprint(load_stage1_config(ROOT / protocol["base_config"]))
        == frozen["base_config_sha256"]
    )
    assert preprocessing_code_fingerprint(ROOT) == frozen["preprocessing_code_sha256"]
    assert training_code_fingerprint(ROOT) == frozen["training_code_sha256"]
    assert (
        sha256_file(PREPARED / "standardizer.pt")
        == registration["data_file_sha256"]["standardizer.pt"]
    )
    flow = torch.load(output / "flow_context_cache.pt", weights_only=True)
    expected = {
        "raw_sha256": sha256_file(ROOT / "data/X-IIoTID/X-IIoTID dataset.csv"),
        **{r: sha256_file(p) for r, p in ROLES.items()},
    }
    assert flow["bindings"] == expected
    replay = torch.load(ROOT / "output/entity_detector/replay_scores.pt", weights_only=True)
    max_error = 0.0
    for seed in (17, 29, 41, 53, 67):
        state = torch.load(output / f"states_seed_{seed}.pt", weights_only=True)
        fingerprints = {
            "detector": sha256_file(ROOT / f"output/entity_detector/detector_seed_{seed}.pt"),
            **{r: sha256_file(p) for r, p in ROLE_PATHS.items()},
        }
        assert state["fingerprints"] == fingerprints
        scores = torch.load(output / f"mechanism_scores_seed_{seed}.pt", weights_only=True)[
            "graph"
        ]["mahalanobis|top3_mean"]
        for band in ("test", "audit"):
            for role in ("calibration", "normal", "attack"):
                actual = scores[role if role == "calibration" else f"{band}_{role}"]
                expected_score = replay[f"{band}#{seed}#{role}"]
                error = float((actual - expected_score).abs().max())
                max_error = max(error, max_error)
                torch.testing.assert_close(actual, expected_score, rtol=1e-12, atol=1e-10)
    detector = FrozenNodeDetector.load(ROOT / "output/entity_detector/detector_seed_17.pt")
    samples = {r: load_graph_samples(p, detector.groups) for r, p in ROLES.items()}
    graphs = [s for role in ROLES for s in samples[role]]
    keys = [(s.metadata["session_key"], s.metadata["window_start_ns"]) for s in graphs]
    assert len(keys) == len(set(keys))
    captures = torch.load(output / "candidate_capture_scores.pt", weights_only=True)
    for saved in captures.values():
        indices = saved["indices"]
        ids = {r: set(v.tolist()) for r, v in indices.items()}
        roles = list(ids)
        for i, role in enumerate(roles):
            assert all(ids[role].isdisjoint(ids[other]) for other in roles[i + 1 :])
        dates = {r: {keys[i][0] for i in values} for r, values in ids.items()}
        assert (dates["train"] | dates["selection"]).isdisjoint(
            dates["normal"] | dates["attack"] | dates["calibration"]
        )
        assert dates["calibration"].isdisjoint(dates["normal"] | dates["attack"])
        for role in ("train", "selection", "calibration", "normal"):
            assert all(graphs[i].label == 0 for i in ids[role])
    commissioning = torch.load(output / "normal_commissioning_scores.pt", weights_only=True)
    for saved in commissioning.values():
        ids = {r: set(v.tolist()) for r, v in saved["target_indices"].items()}
        for role in ("reference", "calibration", "normal"):
            assert all(graphs[i].label == 0 for i in ids[role])
        assert ids["reference"].isdisjoint(ids["calibration"] | ids["normal"] | ids["attack"])
        assert ids["calibration"].isdisjoint(ids["normal"] | ids["attack"])
        for date in {keys[i][0] for i in ids["calibration"]}:
            cutoff = max(keys[i][1] for i in ids["calibration"] if keys[i][0] == date)
            assert all(
                keys[i][1] > cutoff for i in ids["normal"] | ids["attack"] if keys[i][0] == date
            )
    sources = [
        "src/fedxmag/local_support.py",
        "tests/test_local_support.py",
        "scripts/investigate_detection.py",
        "scripts/investigate_context.py",
        "scripts/summarize_root_cause.py",
        "scripts/check_matched_fpr.py",
        "scripts/investigate_information_loss.py",
        "scripts/investigate_generalization.py",
        "scripts/validate_root_cause_candidate.py",
        "scripts/check_capture_feature_shift.py",
        "scripts/check_normal_commissioning.py",
        "scripts/audit_root_cause_artifacts.py",
        "scripts/diagnose_raw_records.py",
        "scripts/diagnose_flow_headroom.py",
        "docs/root_cause_analysis_2026-10-26.md",
    ]
    report = {
        "status": "passed_software_and_artifact_checks_not_scientific_confirmation",
        "created_utc": datetime.now(UTC).isoformat(),
        "checks": {
            "registered_entity_code_unchanged": True,
            "M0_config_training_preprocessing_unchanged": True,
            "source_cache_hashes_match": True,
            "unique_windows": len(keys),
            "baseline_max_absolute_score_difference": max_error,
            "capture_fit_calibration_evaluation_isolation": True,
            "commissioning_reference_and_calibration_normal_only": True,
            "commissioning_evaluation_after_target_calibration": True,
        },
        "inputs": expected,
        "sources": {p: sha256_file(ROOT / p) for p in sources},
        "outputs": {
            p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.suffix in (".pt", ".json")
        },
    }
    write_json(destination, report)
    print(json.dumps(report["checks"]))


if __name__ == "__main__":
    main()
