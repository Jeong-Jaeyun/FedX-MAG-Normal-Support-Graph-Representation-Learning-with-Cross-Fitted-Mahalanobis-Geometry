"""Retrospective diagnostic only: do frozen scores distinguish weak attacks within services?

Target labels select diagnostic cohorts/quantiles, NEVER a deployed model or
checkpoint. Oracle target-normal cutoffs are explicitly not prospective results.
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from fedxmag_next.daily_detector import DailyDeepEnergyDetector
from fedxmag_next.experiments import sha256, write_json
from fedxmag_next.record_data import FEATURES, PROTOCOLS, SERVICES
from fedxmag_next.record_prequential import score_methods

torch.set_num_threads(6)
data_path = Path("New_Model/artifacts/ton_role_coverage_context_v1/records.npz")
model_path = Path("New_Model/runs/round111_ton_role_source/source_detector.pt")
output = Path("New_Model/runs/audit_ton_cohort_energy_v1")
output.mkdir(exist_ok=False)
detector = DailyDeepEnergyDetector.load(model_path)
with np.load(data_path) as payload:
    x = payload["features"]
    label = payload["label"]
    kinds = payload["type"]
    target = payload["observed_at"] >= 1555941600
    selected = detector.state["source_role_indices"]["selection"].numpy()
    rows = np.unique(
        np.r_[
            selected,
            np.flatnonzero(target & ((label == 0) | np.isin(kinds, ["mitm", "ransomware"]))),
        ]
    )
    local_x = x[rows]
    local_label = label[rows]
    local_kind = kinds[rows]
    local_target = target[rows]
write_json(
    output / "registration.json",
    {
        "data_sha256": sha256(data_path),
        "source_detector_sha256": sha256(model_path),
        "purpose": "retrospective diagnostic; target labels never fit or select a detector",
        "limits": "oracle target-normal thresholds are NOT prospective detection results",
    },
)
scores = score_methods(detector, torch.from_numpy(local_x))
proto_columns = [FEATURES.index(f"protocol_{p}") for p in (*PROTOCOLS, "other")]
service_columns = [FEATURES.index(f"service_{p}") for p in (*SERVICES, "other")]
proto = local_x[:, proto_columns].argmax(1)
service = local_x[:, service_columns].argmax(1)
groups = proto * (len(SERVICES) + 1) + service
results = {}
for attack in ["mitm", "ransomware"]:
    result = {}
    for group in np.unique(groups[local_kind == attack]):
        normal = (groups == group) & local_target & (local_label == 0)
        anomaly = (groups == group) & local_target & (local_kind == attack) & (local_label == 1)
        source = (groups == group) & (~local_target)
        if normal.sum() < 20 or not anomaly.any():
            continue
        name = f"{(*PROTOCOLS, 'other')[group // (len(SERVICES) + 1)]}/"
        name += (*SERVICES, "other")[group % (len(SERVICES) + 1)]
        entry = {
            "target_normals": int(normal.sum()),
            "attack_rows": int(anomaly.sum()),
            "source_selection_normals": int(source.sum()),
            "methods": {},
        }
        for method, energy in scores.items():
            q = np.quantile(energy[normal], [0.025, 0.05, 0.5, 0.95, 0.975])
            included = normal | anomaly
            entry["methods"][method] = {
                "target_same_cohort_auroc": float(
                    roc_auc_score(anomaly[included], energy[included])
                ),
                "oracle_target_normal_upper5_tpr": float((energy[anomaly] > q[3]).mean()),
                "oracle_target_normal_lower5_tpr": float((energy[anomaly] < q[1]).mean()),
                "oracle_target_normal_twosided5_tpr": float(
                    ((energy[anomaly] < q[0]) | (energy[anomaly] > q[4])).mean()
                ),
                "target_normal_quantiles": q.tolist(),
                "attack_quantiles": np.quantile(energy[anomaly], [0.05, 0.5, 0.95]).tolist(),
                "source_normal_quantiles": np.quantile(energy[source], [0.05, 0.5, 0.95]).tolist()
                if source.any()
                else [],
            }
        result[name] = entry
        m = entry["methods"]["deep_mahalanobis_m8_ensemble"]
        print(
            attack,
            name,
            entry["attack_rows"],
            "normal",
            entry["target_normals"],
            "AUC",
            round(m["target_same_cohort_auroc"], 3),
            "upper/lower/two",
            *[
                round(m[key], 3)
                for key in [
                    "oracle_target_normal_upper5_tpr",
                    "oracle_target_normal_lower5_tpr",
                    "oracle_target_normal_twosided5_tpr",
                ]
            ],
            flush=True,
        )
    results[attack] = result
write_json(
    output / "report.json",
    {
        "status": "retrospective scalar-score diagnostic only",
        "results": results,
    },
)
torch.save(
    {
        "row_indices": torch.from_numpy(rows),
        "scores": {name: torch.from_numpy(values) for name, values in scores.items()},
    },
    output / "row_scores.pt",
)
