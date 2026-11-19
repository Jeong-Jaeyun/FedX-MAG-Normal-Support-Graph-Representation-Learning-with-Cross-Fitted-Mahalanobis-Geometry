"""Fuse flow and directed-node evidence with an independent normal-only reserve.

The first half of common source-normal windows defines robust scales. A further
300-second purge, including raw flow starts, separates final source calibration.
Target labels never fit fusion parameters. Raw controls receive the same reserve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .experiments import sha256, write_json
from .window_bridge import census, census_sources, complete_role_windows
from .window_replay import run as evaluate_windows


def legacy_node_calibration(run: Path, data: dict, dataset: str, root: Path) -> np.ndarray:
    state = torch.load(run / "node_detector.pt", map_location="cpu", weights_only=True)[
        "flow_encoder"
    ]
    if "source_role_indices" in state:
        rows = state["source_role_indices"]["calibration"].numpy()
    else:
        offset = int(data.get("utc_offset_seconds", 36000))
        date = "2019-04-23" if dataset == "ton" else "2017-07-06"
        cutoff = int(np.datetime64(date, "s").astype(np.int64)) - offset
        normal = np.flatnonzero((data["observed_at"] < cutoff) & (data["label"] == 0))
        normal = normal[np.argsort(data["observed_at"][normal], kind="stable")]
        rows = normal[int(0.85 * len(normal)) :]
    truth = census(census_sources(root, dataset))
    result = np.asarray(complete_role_windows(data["observed_at"], rows, truth))
    registered = json.loads((run / "registration.json").read_text())["normal_windows"][
        "calibration"
    ]
    if registered != len(result):
        raise ValueError("Legacy node calibration chronology cannot be reproduced")
    return result


def run(args: argparse.Namespace) -> None:
    flow, node, output = (
        Path(args.flow).resolve(),
        Path(args.node).resolve(),
        Path(args.output).resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    fr, nr = [json.loads((p / "registration.json").read_text()) for p in (flow, node)]
    if fr["data_sha256"] != nr["data_sha256"] or fr["detector_sha256"] != nr["encoder_sha256"]:
        raise ValueError("Fusion heads must share observations and the frozen flow encoder")
    a, b = [
        torch.load(p / "scores.pt", map_location="cpu", weights_only=True) for p in (flow, node)
    ]
    if not torch.equal(a["minutes"], b["minutes"]) or not torch.equal(a["labels"], b["labels"]):
        raise ValueError("Fusion target populations differ")
    path = Path(fr["args"]["data"])
    with np.load(path) as payload:
        data = {
            key: payload[key]
            for key in ("timestamp", "observed_at", "label", "dataset", "utc_offset_seconds")
            if key in payload
        }
    dataset = str(data.get("dataset", "ton"))
    flow_times = a["methods"]["raw_knn"]["source_calibration_minutes"].numpy()
    node_times = b["methods"]["raw_knn"].get("source_calibration_minutes")
    if node_times is None:
        node_times = legacy_node_calibration(node, data, dataset, Path(args.root).resolve())
    else:
        node_times = node_times.numpy()
    if len(node_times) != len(b["methods"]["raw_knn"]["source_calibration"]):
        raise ValueError("Node source calibration length differs from reproduced minutes")
    common, fi, ni = np.intersect1d(flow_times, node_times, return_indices=True)
    if len(common) < 40:
        raise ValueError("Fewer than 40 independent source-normal windows for fusion/calibration")
    fit = np.arange(len(common) // 2)
    boundary = (int(common[fit[-1]]) + 1) * 60 + 300
    full_minutes = data["observed_at"] // 60
    eligible = []
    for i in range(len(fit), len(common)):
        rows = full_minutes == common[i]
        if common[i] * 60 > boundary and data["timestamp"][rows].min() > boundary:
            eligible.append(i)
    calibration = np.asarray(eligible, dtype=int)
    if len(calibration) < 19:
        raise ValueError("Too few final normal windows after fusion reserve and context purge")
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": dataset,
            "data_sha256": fr["data_sha256"],
            "flow_scores_sha256": sha256(flow / "scores.pt"),
            "node_scores_sha256": sha256(node / "scores.pt"),
            "source_fit_minutes": common[fit].tolist(),
            "final_calibration_minutes": common[calibration].tolist(),
            "fusion": "max of median/q95 normalized heads; fit on source normals only",
            "independence": "300s purge plus raw starts; source roles precede targets",
        },
    )
    archive, scales = {}, {}
    for name in sorted(set(a["methods"]) & set(b["methods"])):
        left, right = a["methods"][name], b["methods"][name]
        lc, rc = left["source_calibration"].numpy()[fi], right["source_calibration"].numpy()[ni]
        transformed, parameter = [], []
        for source_values, target in ((lc, left["energy"]), (rc, right["energy"])):
            center, upper = np.quantile(source_values[fit], [0.5, 0.95])
            scale = max(float(upper - center), 1e-6)
            transformed.append(
                ((source_values - center) / scale, (target.numpy() - center) / scale)
            )
            parameter.append({"center": float(center), "scale": scale})
        scales[name] = parameter
        observed = left["observed"] & right["observed"]
        variants = {
            name: (
                np.maximum(transformed[0][0], transformed[1][0]),
                np.maximum(transformed[0][1], transformed[1][1]),
            ),
            f"flow_only_{name}": transformed[0],
            f"node_only_{name}": transformed[1],
        }
        for variant, (cal, query) in variants.items():
            archive[variant] = {
                "energy": torch.from_numpy(query),
                "observed": observed,
                "source_calibration": torch.from_numpy(cal[calibration]),
                "source_calibration_minutes": torch.from_numpy(common[calibration]),
            }
    torch.save(
        {"methods": archive, "minutes": a["minutes"], "labels": a["labels"], "types": a["types"]},
        output / "scores.pt",
    )
    write_json(output / "fusion_parameters.json", scales)
    evaluate_windows(
        argparse.Namespace(
            run=str(output),
            output=str(output / "evaluation"),
            buffer=args.buffer,
            fpr=0.05,
            history_policy="continuous",
        )
    )
    report = json.loads((output / "evaluation/report.json").read_text())
    write_json(
        output / "report.json",
        {
            **report,
            "unit": "normal_only_flow_node_fusion",
            "normal_fusion_fit": len(fit),
            "normal_final_calibration": len(calibration),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--buffer", type=int, default=20)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
