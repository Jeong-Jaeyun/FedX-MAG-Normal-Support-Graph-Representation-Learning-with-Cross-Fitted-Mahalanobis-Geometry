"""Independently check saved decisions and summarize the registered H1 pilot."""

import json
import math
from pathlib import Path

import numpy as np
import torch

from fedxmag_next.experiments import sha256, write_json
from fedxmag_next.h1_classical_features import SourcePCA
from fedxmag_next.quantum_features import ProjectedQuantumMap

root = Path(__file__).resolve().parents[2]
output = root / "New_Model/reports/h1_pilot_2026-11-15"
output.mkdir(parents=True, exist_ok=False)
summary = {
    "status": "completed",
    "interpretation": "development only; static CPU simulation",
    "analysis_source_sha256": sha256(Path(__file__)),
    "datasets": {},
}


def independent_replay(score, source, minutes, labels, policy):
    history = list(source[-50:])
    queue = []
    choose = np.random.default_rng(731).random(len(minutes)) < 0.1
    result, thresholds = [], []
    for i, minute in enumerate(minutes):
        now = (int(minute) + 1) * 60
        history.extend(value for when, value in queue if when <= now)
        queue = [(when, value) for when, value in queue if when > now]
        history = history[-50:]
        threshold = sorted(history)[math.ceil((len(history) + 1) * 0.95) - 1]
        result.append(score[i] > threshold)
        thresholds.append(threshold)
        if labels[i] == 0 and policy == "immediate":
            history.append(score[i])
        if labels[i] == 0 and policy == "delayed_10pct" and choose[i]:
            queue.append((now + 300, score[i]))
    return np.array(result), np.array(thresholds)


def block_interval(a, b, y, minutes):
    blocks, inverse = np.unique(minutes // 60, return_inverse=True)
    weights = np.random.default_rng(731).multinomial(
        len(blocks), np.full(len(blocks), 1 / len(blocks)), size=2000
    )
    result = {"observed_hour_blocks": len(blocks), "bootstrap_samples": 2000}
    for label, name in [(1, "tpr"), (0, "fpr")]:
        selected = y == label
        numerator = np.bincount(
            inverse[selected],
            weights=a[selected].astype(float) - b[selected],
            minlength=len(blocks),
        )
        denominator = np.bincount(inverse[selected], minlength=len(blocks))
        den = weights @ denominator
        samples = (weights @ numerator)[den > 0] / den[den > 0]
        result[name] = {
            "difference": float(numerator.sum() / denominator.sum()),
            "hour_bootstrap_95": np.quantile(samples, [0.025, 0.975]).tolist(),
        }
    return result


for dataset in ("cic", "xiiotid", "ton"):
    run = root / f"New_Model/runs/round233_h1_{dataset}_pilot_v2"
    report = json.loads((run / "report.json").read_text())
    assert report["status"] == "completed"
    scores = torch.load(run / "scores.pt", weights_only=True)
    decisions = torch.load(run / "decisions.pt", weights_only=True)
    labels, minutes = scores["labels"].numpy(), scores["minutes"].numpy()
    for name, values in scores["methods"].items():
        e, cal = values["energy"].numpy(), values["source_calibration"].numpy()
        for policy in ("fixed", "delayed_10pct", "immediate"):
            f, t = independent_replay(e, cal, minutes, labels, policy)
            saved = decisions[name + "/" + policy]
            assert np.array_equal(f, saved["flags"].numpy())
            assert np.array_equal(t, saved["thresholds"].numpy())
            m = report["methods"][name][policy]
            assert int(f[labels == 1].sum()) == m["tp"]
            assert int(f[labels == 0].sum()) == m["fp"]
    contrasts = {}
    for policy in ("fixed", "delayed_10pct", "immediate"):
        contrasts[policy] = {}
        for candidate, reference in (
            ("quantum_m8", "quantum_noent_m8"),
            ("quantum_m8", "linear_m8"),
            ("quantum_m8", "random_mlp_m8"),
            ("quantum_m8", "frozen_dnn20k_m8"),
            ("raw_plus_quantum_m8", "raw_m8"),
            ("raw_plus_quantum_m8", "raw_plus_quantum_noent_m8"),
        ):
            a = decisions[candidate + "/" + policy]["flags"].numpy()
            b = decisions[reference + "/" + policy]["flags"].numpy()
            comparison = block_interval(a, b, labels, minutes)
            comparison["paired_counts"] = {
                "added_tp": int((a & ~b & (labels == 1)).sum()),
                "lost_tp": int((~a & b & (labels == 1)).sum()),
                "added_fp": int((a & ~b & (labels == 0)).sum()),
                "removed_fp": int((~a & b & (labels == 0)).sum()),
            }
            contrasts[policy][candidate + "_vs_" + reference] = comparison
    # Measurement variability only on source normals; no target-based selection.
    saved = torch.load(run / "source_adapter.pt", weights_only=True)
    data_path = Path(json.loads((run / "registration.json").read_text())["args"]["root"])
    original = {
        "cic": "cic_all_legacy_http_v1",
        "xiiotid": "xiiotid_context_all_v2",
        "ton": "ton_protocol_context_v1",
    }[dataset]
    with np.load(data_path / f"New_Model/artifacts/{original}/records.npz") as payload:
        raw = torch.from_numpy(payload["features"][saved["fit_indices"].numpy()[:32]])
    z = SourcePCA.from_state_dict(saved["adapter"]).transform(raw)
    circuit = ProjectedQuantumMap(seed=17)
    ideal = circuit.transform(z)
    measurement = {}
    for shots in (256, 1024, 4096):
        errors = [
            float((circuit.transform_shots(z, shots=shots, seed=s) - ideal).square().mean().sqrt())
            for s in (101, 103, 107, 109, 113)
        ]
        measurement[str(shots)] = {
            "feature_rmse_replicates": errors,
            "scope": "32 source-normal rows; NOT full-detector finite-shot evaluation",
        }
    summary["datasets"][dataset] = {
        "report_sha256": sha256(run / "report.json"),
        "scores_sha256": sha256(run / "scores.pt"),
        "decisions_sha256": sha256(run / "decisions.pt"),
        "independent_all_decisions_and_thresholds_exact": True,
        "methods": report["methods"],
        "quantum_vs_reference_hour_bootstrap": contrasts,
        "bootstrap_scope": (
            "descriptive paired observed-hour resampling of frozen decisions; "
            "not independent-site inference, not a replay of feedback under resampling, "
            "and not adjusted for multiple comparisons"
        ),
        "source_feature_measurement_audit": measurement,
        "pca_explained_variance_ratio": report["pca_explained_variance_ratio"],
        "representation_audits": report["audits"],
        "target_windows": len(labels),
        "attack_windows": int((labels == 1).sum()),
        "normal_windows": int((labels == 0).sum()),
        "source_fit_rows": len(saved["fit_indices"]),
        "scored_rows": len(saved["combined_indices"]),
        "elapsed_seconds": report["elapsed_seconds"],
    }
    print(dataset, "independent audit passed", len(decisions), flush=True)
write_json(output / "summary.json", summary)
print("saved", output / "summary.json", flush=True)
