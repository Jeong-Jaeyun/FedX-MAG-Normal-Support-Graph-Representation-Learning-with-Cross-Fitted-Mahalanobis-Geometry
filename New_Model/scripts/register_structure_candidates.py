"""Export audited development candidates with their complete initial model state."""

import json
from pathlib import Path

import torch

from fedxmag_next.experiments import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "New_Model/runs"
OUTPUT = ROOT / "New_Model/candidates/prototype_structure_v1"
ENTRIES = (
    ("ton", 173, 170, 165, 39, "raw_mahalanobis_m1", "source", "prototype", "frozen"),
    ("cic", 177, 171, 163, 50, "deep_mahalanobis_m8_ensemble", "source", "prototype", "candidate"),
    (
        "xiiotid",
        175,
        172,
        164,
        39,
        "deep_mahalanobis_m8_ensemble",
        "peer",
        "prototype_risk_fixed",
        "frozen",
    ),
    (
        "xiiotid",
        175,
        172,
        164,
        50,
        "deep_mahalanobis_m8_ensemble",
        "peer",
        "prototype_risk_fixed",
        "frozen",
    ),
)


def report_at(path, capacity):
    return path if capacity == 39 else path / "capacity50"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    records = []
    for (
        dataset,
        final_round,
        prototype_round,
        risk_round,
        capacity,
        base_head,
        scope,
        path,
        feedback,
    ) in ENTRIES:
        suffix = "prototype_calibration" if feedback == "candidate" else "continuous_prototype"
        final = report_at(RUNS / f"round{final_round}_{dataset}_{suffix}", capacity)
        prototype = RUNS / f"round{prototype_round}_{dataset}_prototype_risk"
        risk = RUNS / f"round{risk_round}_{dataset}_causal_risk"
        report = json.loads((final / "report.json").read_text())
        upstream = json.loads((prototype / "report.json").read_text())
        if not report["all_parent_decisions_exact"] or not upstream["all_original_decisions_exact"]:
            raise ValueError("Candidate control replay audit did not pass")
        if not upstream["prototype_midpoint_resume_exact"]:
            raise ValueError("Candidate normal-state restoration audit did not pass")
        if any(v > 1e-5 for v in upstream["projection_restore_scaled_errors"].values()):
            raise ValueError("Candidate actual inference restoration audit did not pass")
        name = (
            f"{base_head}_{'adaptive_' if feedback == 'candidate' else ''}continuous_{path}_{scope}"
        )
        metric = report["methods"][name]
        if metric["fpr"] > 0.05 or metric["coverage"] != 1.0:
            raise ValueError(
                "Registered operating point does not meet its recorded coverage/FPR criterion"
            )
        initial_model = torch.load(prototype / "prototype_risk_detector.pt", weights_only=True)
        source = torch.load(risk / "row_scores.pt", weights_only=True)
        scores = torch.load(final / "scores.pt", weights_only=True)
        packet = {
            "format_version": 1,
            "status": "development_candidate_not_independent_confirmation",
            "dataset": dataset,
            "base_head": base_head,
            "scope": scope,
            "path": path,
            "continuous_filter": True,
            "calibration_feedback": feedback,
            "capacity": capacity,
            "nominal_fpr": 0.05,
            "aggregation": initial_model["protocol"]["aggregation"],
            "prototype_model": initial_model,
            "row_reference": source["row_reference"][base_head].clone(),
            "source_window_calibration": scores["methods"][base_head]["source_calibration"].clone(),
            "source_window_calibration_minutes": scores["methods"][base_head][
                "source_calibration_minutes"
            ].clone(),
            "initial_state": "source-normal seed only; no final target normal/risk state exported",
            "risk_parameters": {"horizon": 300, "decay": 300, "maximum_keys": 5000},
            "risk_clearing": (
                "never clear directly on normal labels; prototype admission after prediction"
            ),
            "data_sha256": initial_model["protocol"]["data_sha256"],
            "source_detector_sha256": initial_model["protocol"]["detector_sha256"],
            "scores_sha256": sha256(final / "scores.pt"),
            "prototype_model_sha256": sha256(prototype / "prototype_risk_detector.pt"),
            "measured_metrics": metric,
            "implementation_modules": ["prototype_guard", "prototype_temporal", "window_bridge"],
            "temporal_gate": "same fixed-feedback base gates as the preserved parent experiment",
        }
        destination = OUTPUT / f"{dataset}_buffer{capacity}.pt"
        torch.save(packet, destination)
        restored = torch.load(destination, weights_only=True)
        for field in (
            "row_reference",
            "source_window_calibration",
            "source_window_calibration_minutes",
        ):
            if not torch.equal(packet[field], restored[field]):
                raise ValueError("Exported candidate altered frozen calibration/reference tensors")
        for old, new in zip(
            initial_model["base_encoder"]["seed_models"],
            restored["prototype_model"]["base_encoder"]["seed_models"],
            strict=True,
        ):
            if any(
                not torch.equal(value, new["model"][key]) for key, value in old["model"].items()
            ):
                raise ValueError("Exported candidate changed learned coefficients")
        records.append(
            {
                "dataset": dataset,
                "capacity": capacity,
                "method": name,
                "model": str(destination),
                "model_sha256": sha256(destination),
                "report": str(final / "report.json"),
                "metrics": metric,
                "matched_frozen_baseline": report["methods"][base_head],
                "uses_neural_energy": base_head.startswith("deep_"),
                "uses_temporal_evidence": path != "prototype",
                "uses_verified_normal_prototypes": True,
                "calibration_feedback": feedback,
            }
        )
    write_json(
        OUTPUT / "registry.json",
        {
            "status": "audited_development_candidates",
            "user_priority": (
                "overall TPR/FPR; retain subtype reporting without a universal subtype gate"
            ),
            "not_a_single_universal_configuration": True,
            "assumptions": [
                "immediate whole-normal verification after prediction",
                "same-network endpoint equality",
                "completed-flow offline timestamp proxy",
                "all supplied data reused for iterative development",
            ],
            "candidates": records,
        },
    )
    print(
        json.dumps(
            [
                {k: r[k] for k in ("dataset", "capacity", "method")}
                | {"tpr": r["metrics"]["tpr"], "fpr": r["metrics"]["fpr"]}
                for r in records
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
