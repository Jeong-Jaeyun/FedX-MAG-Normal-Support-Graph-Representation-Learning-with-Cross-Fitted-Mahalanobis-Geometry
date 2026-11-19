"""Export complete initial cross-view/service states and paired development audits."""

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from fedxmag_next.experiments import sha256, write_json
from fedxmag_next.record_calibration import cluster_difference

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "New_Model/runs"
OUTPUT = ROOT / "New_Model/candidates/prototype_structure_v2"


def at_capacity(path, capacity):
    return path if capacity == 39 else path / "capacity50"


def audit(candidate_root, candidate_name, baseline_root, baseline_name, capacity):
    current = torch.load(candidate_root / "scores.pt", weights_only=True)
    previous = torch.load(baseline_root / "scores.pt", weights_only=True)
    for field in ("minutes", "labels"):
        if not torch.equal(current[field], previous[field]):
            raise ValueError("Candidate comparison changed its windows or truth")
    if current["types"] != previous["types"]:
        raise ValueError("Candidate comparison changed its subtype census")
    flags = []
    for folder, name in ((candidate_root, candidate_name), (baseline_root, baseline_name)):
        decisions = torch.load(folder / "evaluation/decisions.pt", weights_only=True)[name]
        flags.append(tuple(decisions[key].numpy() for key in ("flags", "valid")))
    if not np.array_equal(flags[0][1], flags[1][1]) or not flags[0][1].all():
        raise ValueError("Paired development comparison must retain full common coverage")
    labels = current["labels"].numpy()
    return {
        "status": "paired_reused_development_data_not_independent_confirmation",
        "candidate": candidate_name,
        "baseline": baseline_name,
        "capacity": capacity,
        "all_minutes_labels_types_and_coverage_exact": True,
        "counts": {
            kind: {
                "total": int((labels == value).sum()),
                "candidate": int(flags[0][0][labels == value].sum()),
                "baseline": int(flags[1][0][labels == value].sum()),
                "gained": int(((labels == value) & flags[0][0] & ~flags[1][0]).sum()),
                "lost": int(((labels == value) & ~flags[0][0] & flags[1][0]).sum()),
            }
            for kind, value in (("attack", 1), ("normal", 0))
        },
        "hour_block_difference_CI": cluster_difference(
            *flags, labels, current["minutes"].numpy() // 60
        ),
        "candidate_scores_sha256": sha256(candidate_root / "scores.pt"),
        "baseline_scores_sha256": sha256(baseline_root / "scores.pt"),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    old = json.loads(
        (ROOT / "New_Model/candidates/prototype_structure_v1/registry.json").read_text()
    )
    entries = []
    for record in old["candidates"]:
        if record["dataset"] == "cic":
            continue
        record = {**record}
        source = Path(record["model"])
        if sha256(source) != record["model_sha256"]:
            raise ValueError("Previously audited candidate packet changed")
        destination = OUTPUT / source.name
        shutil.copyfile(source, destination)
        record["model"] = str(destination)
        entries.append(record)
    for (
        dataset,
        capacity,
        base_head,
        scope,
        normal_view,
        feedback,
        final_name,
        prototype_round,
        risk_round,
    ) in (
        (
            "cic",
            39,
            "deep_mahalanobis_m1_ensemble",
            "source",
            "raw",
            "candidate",
            "round180_cic_cross_view",
            171,
            163,
        ),
        (
            "cic",
            50,
            "deep_mahalanobis_m8_ensemble",
            "source",
            "raw",
            "candidate",
            "round180_cic_cross_view",
            171,
            163,
        ),
        (
            "xiiotid",
            50,
            "deep_mahalanobis_m8_ensemble",
            "peer",
            "deep",
            "frozen",
            "round184_xiiotid_service_risk",
            172,
            164,
        ),
    ):
        final = at_capacity(RUNS / final_name, capacity)
        service = "service_risk" in final_name
        tag = ("rawview_" if normal_view == "raw" else "") + ("service_" if service else "")
        feedback_tag = "adaptive_" if feedback == "candidate" else ""
        method = f"{base_head}_{tag}{feedback_tag}continuous_prototype_risk_fixed_{scope}"
        report = json.loads((final / "report.json").read_text())
        metric = report["methods"][method]
        if not report["all_parent_decisions_exact"] or not report["matched_view_recomputed_exact"]:
            raise ValueError("Matched control trajectory did not reproduce exactly")
        if metric["coverage"] != 1 or metric["fpr"] > 0.05:
            raise ValueError("Operating point fails recorded coverage/FPR criterion")
        prototype = RUNS / f"round{prototype_round}_{dataset}_prototype_risk"
        risk = RUNS / f"round{risk_round}_{dataset}_causal_risk"
        upstream = json.loads((prototype / "report.json").read_text())
        if not upstream["prototype_midpoint_resume_exact"]:
            raise ValueError("Actual normal memory checkpoint audit failed")
        if not all(
            item["checkpoint_resumed_exact"]
            for name, item in report["path_diagnostics"].items()
            if "prototype_risk_fixed" in name
        ):
            raise ValueError("Actual temporal checkpoint audit failed")
        model = torch.load(prototype / "prototype_risk_detector.pt", weights_only=True)
        cache = torch.load(risk / "row_scores.pt", weights_only=True)
        scores = torch.load(final / "scores.pt", weights_only=True)
        packet = {
            "format_version": 2,
            "status": "development_candidate_not_independent_confirmation",
            "dataset": dataset,
            "base_head": base_head,
            "normal_metric": normal_view,
            "scope": scope,
            "risk_key_fields": [
                "source",
                "destination",
                "destination_port",
                "protocol",
                "available_dst_port",
            ]
            if service
            else ["source"],
            "path": "prototype_risk_fixed",
            "continuous_filter": True,
            "calibration_feedback": feedback,
            "capacity": capacity,
            "nominal_fpr": 0.05,
            "aggregation": model["protocol"]["aggregation"],
            "prototype_model": model,
            "row_reference": cache["row_reference"][base_head].clone(),
            "source_window_calibration": scores["methods"][base_head]["source_calibration"].clone(),
            "source_window_calibration_minutes": scores["methods"][base_head][
                "source_calibration_minutes"
            ].clone(),
            "initial_state": "source-normal seed only; no final target normal/risk state exported",
            "risk_parameters": {"horizon": 300, "decay": 300, "maximum_keys": 5000},
            "risk_clearing": (
                "never clear directly on normal labels; prototype admission after prediction"
            ),
            "temporal_gate": (
                "fixed original-feedback base gates; "
                "candidate feedback changes final thresholds only"
            ),
            "data_sha256": model["protocol"]["data_sha256"],
            "source_detector_sha256": model["protocol"]["detector_sha256"],
            "scores_sha256": sha256(final / "scores.pt"),
            "prototype_model_sha256": sha256(prototype / "prototype_risk_detector.pt"),
            "measured_metrics": metric,
            "implementation_modules": ["prototype_guard", "prototype_temporal", "window_bridge"]
            + (["service_risk_keys"] if service else []),
        }
        destination = OUTPUT / f"{dataset}_{'service_' if service else ''}buffer{capacity}.pt"
        torch.save(packet, destination)
        restored = torch.load(destination, weights_only=True)
        for field in (
            "row_reference",
            "source_window_calibration",
            "source_window_calibration_minutes",
        ):
            if not torch.equal(packet[field], restored[field]):
                raise ValueError("Candidate export changed a frozen reference")
        for previous, actual in zip(
            model["base_encoder"]["seed_models"],
            restored["prototype_model"]["base_encoder"]["seed_models"],
            strict=True,
        ):
            if any(
                not torch.equal(value, actual["model"][key])
                for key, value in previous["model"].items()
            ):
                raise ValueError("Candidate export changed learned coefficients")
        baseline_root = at_capacity(
            RUNS
            / (
                "round175_xiiotid_continuous_prototype"
                if service
                else "round177_cic_prototype_calibration"
            ),
            capacity,
        )
        baseline_name = (
            f"{base_head}_continuous_prototype_risk_fixed_{scope}"
            if service
            else "deep_mahalanobis_m8_ensemble_adaptive_continuous_prototype_source"
        )
        comparison = audit(final, method, baseline_root, baseline_name, capacity)
        write_json(OUTPUT / f"{destination.stem}_paired.json", comparison)
        if not service:
            matched = f"{base_head}_adaptive_continuous_prototype_risk_fixed_{scope}"
            write_json(
                OUTPUT / f"{destination.stem}_matched_view_paired.json",
                audit(final, method, final, matched, capacity),
            )
        entries.append(
            {
                "dataset": dataset,
                "capacity": capacity,
                "method": method,
                "model": str(destination),
                "model_sha256": sha256(destination),
                "report": str(final / "report.json"),
                "metrics": metric,
                "matched_frozen_baseline": report["methods"][base_head],
                "uses_neural_energy": True,
                "normal_metric": normal_view,
                "uses_temporal_evidence": True,
                "uses_verified_normal_prototypes": True,
                "calibration_feedback": feedback,
                "paired_comparison": comparison,
            }
        )
    write_json(
        OUTPUT / "registry.json",
        {**old, "candidates": entries, "supersedes": "prototype_structure_v1; old files retained"},
    )
    print(
        json.dumps(
            [
                {
                    "dataset": e["dataset"],
                    "capacity": e["capacity"],
                    "method": e["method"],
                    "tpr": e["metrics"]["tpr"],
                    "fpr": e["metrics"]["fpr"],
                }
                for e in entries
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
