"""Post-hoc attribution from archived scores; does not refit or alter validation."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedxmag.m1.covariance import (  # noqa: E402
    differentiable_shape_terms,
    estimate_covariance,
    mahalanobis_energy,
)


def main() -> None:
    base = ROOT / "output/cic_ids2017_validation_20261101"
    archive = torch.load(base / "scores.pt", weights_only=True)
    records = torch.load(base / "views/window_graphs.pt", weights_only=True)["records"]
    metadata = [json.loads(item["metadata_json"]) for item in records]
    scores, thresholds = archive["window"], archive["primary_thresholds"]
    positive = archive["positive"].bool()
    common = torch.stack([torch.isfinite(thresholds[k])
                          for k in ("detector", "raw_node", "flow")]).all(0)
    alerts = {k: scores[k] > thresholds[k] for k in thresholds}

    def rate(values: torch.Tensor, mask: torch.Tensor) -> float | None:
        return float(values[mask].double().mean()) if mask.any() else None

    normal = common & ~positive
    result = {
        "status": "exploratory_archived_score_attribution_no_new_validation",
        "normal_windows": int(normal.sum()),
        "channel_fpr": {k: rate(alerts[k], normal)
                        for k in ("detector", "raw_node", "flow")},
        "attack_types": {},
    }
    # For F=max(N/a,V/b), max(N/F) and max(V/F) recover the scales
    # if both branches are active somewhere. Check reconstruction before interpreting.
    nonzero = scores["detector"] > 0
    inferred_scales = {k: float((scores[k][nonzero] / scores["detector"][nonzero]).max())
                       for k in ("raw_node", "flow")}
    reconstructed = torch.maximum(scores["raw_node"] / inferred_scales["raw_node"],
                                  scores["flow"] / inferred_scales["flow"])
    result["fusion_scale_attribution"] = {
        "inferred_scales": inferred_scales,
        "max_reconstruction_error": float((reconstructed - scores["detector"]).abs().max()),
        "flow_at_least_raw_fraction": float((
            scores["flow"] / inferred_scales["flow"] >=
            scores["raw_node"] / inferred_scales["raw_node"]).double().mean()),
        "fused_and_flow_alert_disagreements": int((alerts["detector"][common] !=
                                                   alerts["flow"][common]).sum()),
    }
    for kind in sorted({k for item in metadata for k in item["attack_types"]}):
        mask = torch.tensor([kind in item["attack_types"] for item in metadata]) & common
        result["attack_types"][kind] = {
            "windows": int(mask.sum()),
            "tpr": {k: rate(alerts[k], mask) for k in ("detector", "raw_node", "flow")},
            "raw_alert_but_fused_miss": int((mask & alerts["raw_node"]
                                             & ~alerts["detector"]).sum()),
            "fused_score_median": float(scores["detector"][mask].median()),
            "fused_threshold_median": float(thresholds["detector"][mask].median()),
        }
    calibration = archive["monday_calibration_detector"].sort().values
    threshold = calibration[math.ceil((len(calibration) + 1) * .95) - 1]
    result["static_source_threshold_counterfactual"] = {
        "note": "Same common windows; threshold fixed on Monday, no target labels used to fit it",
        "threshold": float(threshold),
        "fpr": rate(scores["detector"] > threshold, normal),
        "tpr": rate(scores["detector"] > threshold, common & positive),
    }
    # Numerical mechanism checks, not evidence that these mechanisms caused a run.
    r = torch.randn(80, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(42))
    q = torch.randn(50, 12, dtype=torch.float64, generator=torch.Generator().manual_seed(43))
    original = mahalanobis_energy(q, estimate_covariance(r, "ledoit_wolf"))
    result["uniform_scale_invariance"] = {
        str(c): float((original - mahalanobis_energy(
            c * q, estimate_covariance(c * r, "ledoit_wolf"))).abs().max())
        for c in (.1, 10.)
    }
    rank_one = torch.diag(torch.tensor([32.] + [0.] * 31, dtype=torch.float64))
    identity = torch.eye(32, dtype=torch.float64)
    result["rank_guard_counterexample"] = []
    for alpha in (.5, .9):
        log_condition, rank = differentiable_shape_terms(
            (1 - alpha) * rank_one + (alpha + 1e-6) * identity)
        result["rank_guard_counterexample"].append({
            "raw_rank": 1, "alpha": alpha, "shrunk_effective_rank": float(rank),
            "condition": float(log_condition.exp()),
            "rank_min8_penalty": float(torch.relu(8 - rank).square()),
        })
    destination = ROOT / "output/failure_attribution_20261102.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
