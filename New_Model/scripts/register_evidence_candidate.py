"""Preserve the first X repeated-evidence95%/5% candidate as an initial packet."""

import json
import shutil
from pathlib import Path

import torch
from register_cross_view_candidates import audit

from fedxmag_next.experiments import sha256, write_json

root = Path(__file__).resolve().parents[2]
previous = root / "New_Model/candidates/prototype_structure_v2"
output = root / "New_Model/candidates/prototype_structure_v3"
output.mkdir(parents=True, exist_ok=False)
registry = json.loads((previous / "registry.json").read_text())
records = []
for record in registry["candidates"]:
    old_path = Path(record["model"])
    if sha256(old_path) != record["model_sha256"]:
        raise ValueError("Prior audited candidate state changed")
    new_path = output / old_path.name
    shutil.copyfile(old_path, new_path)
    records.append({**record, "model": str(new_path)})
origin = root / "New_Model/runs/round212_xiiotid_service_evidence/capacity50"
report = json.loads((origin / "report.json").read_text())
method = "deep_mahalanobis_m8_ensemble_service_evidence_sum_weak_peer_frozen"
metric = report["methods"][method]
if not report["all_matched_parent_trajectories_and_decisions_exact"]:
    raise ValueError("Repeated-evidence controls failed")
if metric["coverage"] != 1 or metric["tpr"] < 0.95 or metric["fpr"] > 0.05:
    raise ValueError("Repeated-evidence candidate fails the explicit95%/5% target")
if not all(item["midpoint_disk_restore_exact"] for item in report["evidence_diagnostics"].values()):
    raise ValueError("Actual repeated-evidence checkpoint audit failed")
packet = torch.load(previous / "xiiotid_service_buffer50.pt", weights_only=True)
scores = torch.load(origin / "scores.pt", weights_only=True)
for field in ("source_window_calibration", "source_window_calibration_minutes"):
    source_field = field.replace("source_window_", "source_")
    if not torch.equal(packet[field], scores["methods"][packet["base_head"]][source_field]):
        raise ValueError("Repeated-evidence source calibration changed")
packet.update(
    {
        "format_version": 3,
        "evidence_rule": {
            "mode": "sum",
            "initiation": "weak",
            "contribution": "strongest_original_per_key_per_window",
        },
        "measured_metrics": metric,
        "scores_sha256": sha256(origin / "scores.pt"),
        "implementation_modules": packet["implementation_modules"]
        + ["weak_evidence", "candidate_stream"],
        "temporal_gate": (
            "novel original filtered row energy above source-selection q95; no window alarm gate"
        ),
        "risk_clearing": "no direct normal-label clearing; propagate only prior original evidence",
    }
)
destination = output / "xiiotid_evidence_buffer50.pt"
torch.save(packet, destination)
restored = torch.load(destination, weights_only=True)
if not torch.equal(packet["row_reference"], restored["row_reference"]):
    raise ValueError("Repeated-evidence export changed a row reference")
for old, new in zip(
    packet["prototype_model"]["base_encoder"]["seed_models"],
    restored["prototype_model"]["base_encoder"]["seed_models"],
    strict=True,
):
    if any(not torch.equal(value, new["model"][key]) for key, value in old["model"].items()):
        raise ValueError("Repeated-evidence export changed learned coefficients")
comparisons = {}
for label, folder, baseline in (
    (
        "previous_service",
        "round184_xiiotid_service_risk",
        "deep_mahalanobis_m8_ensemble_service_continuous_prototype_risk_fixed_peer",
    ),
    (
        "endpoint_evidence",
        "round206_xiiotid_weak_evidence",
        "deep_mahalanobis_m8_ensemble_evidence_sum_weak_peer_frozen",
    ),
):
    comparisons[label] = audit(
        origin, method, root / "New_Model/runs" / folder / "capacity50", baseline, 50
    )
    write_json(output / f"xiiotid_evidence50_{label}_paired.json", comparisons[label])
records.append(
    {
        "dataset": "xiiotid",
        "capacity": 50,
        "method": method,
        "model": str(destination),
        "model_sha256": sha256(destination),
        "report": str(origin / "report.json"),
        "metrics": metric,
        "uses_neural_energy": True,
        "normal_metric": "deep",
        "uses_temporal_evidence": True,
        "uses_verified_normal_prototypes": True,
        "calibration_feedback": "frozen",
        "evidence_rule": packet["evidence_rule"],
        "paired_comparisons": comparisons,
    }
)
write_json(
    output / "registry.json",
    {
        **registry,
        "candidates": records,
        "user_success_criterion": "ALL THREE datasets TPR>=.95 and FPR<=.05",
        "goal_status": "CIC remains below target; exported X candidate needs integrated API replay",
        "supersedes": "prototype_structure_v2; all older files retained",
    },
)
print(
    json.dumps(
        {
            "model": str(destination),
            "tpr": metric["tpr"],
            "fpr": metric["fpr"],
            "paired": comparisons,
        },
        indent=2,
    )
)
