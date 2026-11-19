"""Symmetric joint-rank test on flow/node energies and past verified normals.

Ranks are recomputed symmetrically for the normal reserve and the single query.
No query label or query-dependent geometry/weight fitting is used. This removes
fixed source-scale dominance; temporal exchangeability is not assumed in reports.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .experiments import sha256, write_json
from .record_calibration import cluster_difference
from .window_fusion import legacy_node_calibration


def joint_rank_pvalue(
    reference: np.ndarray, query: np.ndarray, *, pooling: str = "any"
) -> float:
    values = np.vstack((reference, query))
    if not np.isfinite(values).all():
        raise ValueError("Joint rank inputs must be finite")
    ranks = np.empty(values.shape, dtype=int)
    for column in range(values.shape[1]):
        ordered = np.sort(values[:, column])
        ranks[:, column] = len(values) - np.searchsorted(ordered, values[:, column], side="left")
    if pooling == "any":
        extreme = ranks.min(axis=1)
        return float(np.mean(extreme <= extreme[-1]))
    surprise = -np.log(ranks / len(values))
    if pooling == "mean":
        combined = surprise.mean(axis=1)
    elif pooling == "median":
        combined = np.median(surprise, axis=1)
    elif pooling == "all":
        combined = surprise.min(axis=1)
    else:
        raise ValueError("Unknown symmetric rank pooling rule")
    return float(np.mean(combined >= combined[-1]))


def run(args: argparse.Namespace) -> None:
    flow, node, output = (
        Path(args.flow).resolve(),
        Path(args.node).resolve(),
        Path(args.output).resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    fr, nr = [json.loads((p / "registration.json").read_text()) for p in (flow, node)]
    if fr["data_sha256"] != nr["data_sha256"] or fr["detector_sha256"] != nr["encoder_sha256"]:
        raise ValueError("Joint heads must share data and frozen flow encoder")
    a, b = [
        torch.load(p / "scores.pt", map_location="cpu", weights_only=True) for p in (flow, node)
    ]
    for key in ("minutes", "labels"):
        if not torch.equal(a[key], b[key]):
            raise ValueError("Joint target populations differ")
    with np.load(fr["args"]["data"]) as payload:
        data = {
            key: payload[key]
            for key in ("timestamp", "observed_at", "label", "dataset", "utc_offset_seconds")
            if key in payload
        }
    dataset = str(data.get("dataset", "ton"))
    ft = a["methods"]["raw_knn"]["source_calibration_minutes"].numpy()
    nt = b["methods"]["raw_knn"].get("source_calibration_minutes")
    nt = (
        nt.numpy()
        if nt is not None
        else legacy_node_calibration(node, data, dataset, Path(args.root).resolve())
    )
    common, fi, ni = np.intersect1d(ft, nt, return_indices=True)
    if len(nt) != len(b["methods"]["raw_knn"]["source_calibration"]):
        raise ValueError("Legacy calibration minute reconstruction differs")
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": dataset,
            "flow_scores_sha256": sha256(flow / "scores.pt"),
            "node_scores_sha256": sha256(node / "scores.pt"),
            "source_normal_minutes": common.tolist(),
            "fit": "none; symmetric upper-tail ranks of normal reserve plus one query",
            "feedback": "same observed whole-normal windows, admitted only after prediction",
            "pvalue_contract": "exchangeable symmetry; temporal FPR measured separately",
        },
    )
    labels, minutes = a["labels"].numpy(), a["minutes"].numpy()
    offset = int(data.get("utc_offset_seconds", 36000))
    captures = (minutes * 60 + offset) // 86400
    results, archive = {}, {}
    for name in sorted(set(a["methods"]) & set(b["methods"])):
        left, right = a["methods"][name], b["methods"][name]
        calibration = np.column_stack(
            (left["source_calibration"].numpy()[fi], right["source_calibration"].numpy()[ni])
        )
        query = np.column_stack((left["energy"].numpy(), right["energy"].numpy()))
        observed = (left["observed"] & right["observed"]).numpy()
        for suffix, columns in (("", [0, 1]), ("flow_only_", [0]), ("node_only_", [1])):
            method = suffix + name
            history = deque(calibration[-args.buffer :, columns].tolist(), maxlen=args.buffer)
            pvalues, valid = np.ones(len(query)), np.zeros(len(query), dtype=bool)
            for i in range(len(query)):
                if observed[i] and len(history) >= math.ceil(1 / 0.05) - 1:
                    pvalues[i] = joint_rank_pvalue(np.asarray(history), query[i, columns])
                    valid[i] = True
                if observed[i] and labels[i] == 0:
                    history.append(query[i, columns].tolist())
            flags = valid & (pvalues <= 0.05)
            normal, attack = valid & (labels == 0), valid & (labels == 1)
            result = {
                "tpr": float(flags[attack].mean()) if attack.any() else None,
                "fpr": float(flags[normal].mean()) if normal.any() else None,
                "coverage": float(valid.mean()),
                "normal_windows": int(normal.sum()),
                "attack_windows": int(attack.sum()),
                "end_to_end_attack_recall": float(flags[labels == 1].mean()),
                "auroc_of_adaptive_scores": float(
                    roc_auc_score(labels[valid], -np.log(pvalues[valid]))
                )
                if normal.any() and attack.any()
                else None,
            }
            result["per_type_tpr"] = {}
            for kind in sorted({kind for group in a["types"] for kind in group}):
                available = np.asarray([kind in group for group in a["types"]])
                supported = available & valid
                result["per_type_tpr"][kind] = {
                    "available_windows": int(available.sum()),
                    "evaluated_windows": int(supported.sum()),
                    "tpr": float(flags[supported].mean()) if supported.any() else None,
                }
            result["per_capture"] = {}
            for capture in np.unique(captures):
                group = captures == capture
                ni_mask, ai_mask = normal & group, attack & group
                result["per_capture"][str(int(capture))] = {
                    "normal_windows": int(ni_mask.sum()),
                    "attack_windows": int(ai_mask.sum()),
                    "fpr": float(flags[ni_mask].mean()) if ni_mask.any() else None,
                    "tpr": float(flags[ai_mask].mean()) if ai_mask.any() else None,
                }
            results[method] = result
            archive[method] = {
                "flags": torch.from_numpy(flags),
                "valid": torch.from_numpy(valid),
                "pvalue": torch.from_numpy(pvalues),
            }
            print(method, {key: result[key] for key in ("tpr", "fpr", "coverage")}, flush=True)
    differences = {}
    for method in results:
        if method.startswith("deep_"):
            differences[method] = {}
            for baseline in ("raw_knn", "flow_only_raw_knn", "node_only_raw_knn"):
                differences[method][baseline] = cluster_difference(
                    (archive[method]["flags"].numpy(), archive[method]["valid"].numpy()),
                    (archive[baseline]["flags"].numpy(), archive[baseline]["valid"].numpy()),
                    labels,
                    minutes // 60,
                )
    write_json(
        output / "report.json",
        {
            "status": "symmetric_joint_rank_development",
            "dataset": dataset,
            "methods": results,
            "hour_bootstrap_vs_controls": differences,
        },
    )
    torch.save(archive, output / "decisions.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--buffer", type=int, default=50)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
