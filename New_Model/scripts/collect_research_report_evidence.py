"""Collect compact, attributable facts for the consolidated Korean report."""

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/research_review_2026-11-09"


def read(relative):
    path = ROOT / relative
    data = json.loads(path.read_text(encoding="utf-8"))
    return data, hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    points = (
        ("ton_exported", "runs/round173_ton_continuous_prototype/report.json",
         "raw_mahalanobis_m1_continuous_prototype_source"),
        ("ton_measured", "runs/round210_ton_service_evidence/report.json",
         "raw_mahalanobis_m1_service_evidence_peak_weak_source_frozen"),
        ("x_exported", "runs/round212_xiiotid_service_evidence/capacity50/report.json",
         "deep_mahalanobis_m8_ensemble_service_evidence_sum_weak_peer_frozen"),
        ("cic_exported", "runs/round180_cic_cross_view/report.json",
         "deep_mahalanobis_m1_ensemble_rawview_adaptive_continuous_prototype_risk_fixed_source"),
        ("cic_legacy_best", "runs/round225_cic_score_evidence_factorial/capacity50/report.json",
         "deep_mahalanobis_m8_ensemble_normal_units_service_deep8_sum_gated_endpoint_candidate"),
        ("cic_full_http_best", "runs/round219_cic_current_normal_view/report.json",
         "deep_mahalanobis_m1_ensemble_current_guard_service_sum_weak_source_candidate"),
        ("cic_full_http_static", "runs/round219_cic_current_normal_view/report.json",
         "deep_mahalanobis_m1_ensemble_sparse_top3"),
        ("ton_heterogeneous_mean", "runs/round230_ton_heterogeneous_consensus/capacity50/report.json",
         "heterogeneous_neural_gaussian_mean_component_native"),
        ("ton_heterogeneous_median", "runs/round230_ton_heterogeneous_consensus/capacity50/report.json",
         "heterogeneous_neural_forest_gaussian_median_component_native"),
    )
    evidence = {"as_of_utc": "2026-11-09", "points": {}, "models": {}}
    for identity, relative, method in points:
        data, digest = read(relative)
        evidence["points"][identity] = {
            "source": relative, "source_sha256": digest, "method": method,
            "metrics": data["methods"][method],
        }
    for name, relative in (
        ("ton", "runs/round170_ton_prototype_risk/prototype_risk_detector.pt"),
        ("xiiotid", "runs/round172_xiiotid_prototype_risk/prototype_risk_detector.pt"),
        ("cic", "runs/round186_cic_packet_http_source/source_detector.pt"),
    ):
        state = torch.load(ROOT / relative, weights_only=True)
        state = state.get("base_encoder", state)
        models = []
        for item in state["seed_models"]:
            history = item["history"]
            losses = [h for h in history if "selection_synthetic_bce" in h]
            best = min(losses, key=lambda h: h["selection_synthetic_bce"]) if losses else None
            models.append({
                "seed": item["seed"], "epochs_run": max(h["epoch"] for h in history),
                "best_source_selection_epoch": None if best is None else best["epoch"],
            })
        evidence["models"][name] = {
            "source": relative, "encoder_kind": state.get("encoder_kind", "dependency"),
            "input_dimension": len(state["feature_names"]), "models": models,
            "source_role_counts": {k: len(v) for k, v in state["source_role_indices"].items()},
        }
    for identity, relative in (
        ("validated_replays", "candidates/prototype_structure_v3/validated_replays.json"),
        ("hard_votes", "runs/round232_heterogeneous_hard_votes/report.json"),
    ):
        data, digest = read(relative)
        evidence[identity] = {"source": relative, "source_sha256": digest, "data": data}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "evidence.json"
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"evidence": str(path), "points": len(evidence["points"]),
                      "model_dimensions": {k: v["input_dimension"] for k, v in evidence["models"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    run()
