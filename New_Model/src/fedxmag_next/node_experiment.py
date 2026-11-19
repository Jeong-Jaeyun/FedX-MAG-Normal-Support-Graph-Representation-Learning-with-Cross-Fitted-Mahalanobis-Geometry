"""Frozen DNN flow states pooled into anonymous directed node support geometries.

Normal whole source windows fit node geometry. Independent source selection
normalizes seeds; whole normal calibration windows determine final graph alarms.
All original target-window labels and all failed types remain in evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .daily_detector import DailyDeepEnergyDetector
from .experiments import knn_scores, sha256, write_json
from .geometry import ConditionalMahalanobis
from .window_bridge import census, census_sources
from .window_replay import run as evaluate_windows


def pool_nodes(
    values: torch.Tensor, outgoing: torch.Tensor, incoming: torch.Tensor, nodes: int
) -> torch.Tensor:
    result = []
    for index in (outgoing, incoming):
        count = torch.bincount(index, minlength=nodes)
        total = torch.zeros(nodes, values.shape[1], dtype=values.dtype)
        total.index_add_(0, index, values)
        maximum = torch.full_like(total, -torch.inf)
        maximum.scatter_reduce_(
            0, index[:, None].expand_as(values), values, reduce="amax", include_self=True
        )
        maximum[count == 0] = 0
        result.extend((total / count.clamp_min(1)[:, None], maximum, count.log1p()[:, None]))
    return torch.cat(result, dim=1)


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    output, path, model_path = (
        Path(args.output).resolve(),
        Path(args.data).resolve(),
        Path(args.detector).resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    with np.load(path) as payload:
        data = {key: payload[key] for key in payload.files}
    with np.load(args.keys) as payload:
        keys = {key: payload[key] for key in payload.files}
    if not np.array_equal(data["signature"], keys["signature"]):
        raise ValueError("Entity keys refer to different row observations")
    detector = DailyDeepEnergyDetector.load(model_path)
    origin = json.loads((model_path.parent / "registration.json").read_text())
    if origin["data_sha256"] != sha256(path):
        raise ValueError("Frozen flow encoder belongs to a different dataset")
    dataset = str(data.get("dataset", "ton"))
    offset = int(data.get("utc_offset_seconds", 36000))
    tables = census_sources(Path(args.root).resolve(), dataset)
    truth = census(tables)
    ts, labels = data["observed_at"], data["label"]
    if dataset == "xiiotid":
        days = np.unique((data["timestamp"] + offset) // 86400)
        cutoff, end = (
            int(days[int(0.6 * len(days))]) * 86400 - offset,
            (int(days[-1]) + 1) * 86400 - offset,
        )
    else:
        bounds = ("2019-04-23", "2019-05-01") if dataset == "ton" else ("2017-07-06", "2017-07-08")
        cutoff, end = [int(np.datetime64(day, "s").astype(np.int64)) - offset for day in bounds]
    if detector.state["latest_reference_observed_at"] >= cutoff:
        raise ValueError("Frozen encoder crossed first target time")
    if "source_role_indices" in detector.state:
        roles = {name: rows.numpy() for name, rows in detector.state["source_role_indices"].items()}
    else:
        normal = np.flatnonzero((ts < cutoff) & (labels == 0))
        normal = normal[np.argsort(ts[normal], kind="stable")]
        a, b = int(0.7 * len(normal)), int(0.85 * len(normal))
        roles = {"train": normal[:a], "selection": normal[a:b], "calibration": normal[b:]}
    if any(np.any(labels[rows] != 0) or np.any(ts[rows] >= cutoff) for rows in roles.values()):
        raise ValueError("Non-normal or future observation entered source roles")
    if sum(len(rows) for rows in roles.values()) != len(
        np.unique(np.concatenate(list(roles.values())))
    ):
        raise ValueError("Source roles overlap")
    minutes = ts // 60
    compound = np.empty(2 * len(ts), dtype=[("minute", "<i8"), ("node", "<u8")])
    compound["minute"] = np.tile(minutes, 2)
    compound["node"] = np.concatenate((keys["source"], keys["destination"]))
    unique, inverse = np.unique(compound, return_inverse=True)
    outgoing, incoming = [torch.from_numpy(part.copy()) for part in np.split(inverse, 2)]
    node_minutes = unique["minute"]
    row_groups = {}
    order = np.argsort(minutes, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(minutes[order]) != 0])
    for start, stop in zip(starts, np.r_[starts[1:], len(order)], strict=True):
        row_groups[int(minutes[order[start]])] = order[start:stop]
    role_windows = {}
    for name, rows in roles.items():
        member = np.zeros(len(ts), dtype=bool)
        member[rows] = True
        role_windows[name] = [
            minute
            for minute, group in row_groups.items()
            if truth[minute]["label"] == 0 and member[group].all()
        ]
    node_roles = {
        name: np.flatnonzero(np.isin(node_minutes, windows))
        for name, windows in role_windows.items()
    }
    if min(len(node_roles["train"]), len(node_roles["selection"])) < 32:
        raise ValueError("Too few independent whole-normal source nodes")
    reference = np.random.default_rng(17).permutation(node_roles["train"])[: args.max_reference]
    selection = node_roles["selection"]
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": dataset,
            "data_sha256": sha256(path),
            "encoder_sha256": sha256(model_path),
            "keys_sha256": sha256(Path(args.keys)),
            "census_sha256": {p.name: sha256(p) for p in tables},
            "normal_windows": {name: len(windows) for name, windows in role_windows.items()},
            "normal_node_roles": {name: len(rows) for name, rows in node_roles.items()},
            "query_contract": (
                "completed-minute outgoing/incoming mean/max/count; no identity input"
            ),
        },
    )
    print(
        {"phase": "node_support_fit", "normal_nodes": len(reference), "nodes": len(unique)},
        flush=True,
    )
    values = detector.normalize_features(torch.from_numpy(data["features"]))
    scores, artifact = {}, {"flow_encoder": detector.state, "node_models": {}}

    def standardize(pooled):
        mean = pooled[reference].mean(0)
        scale = pooled[reference].std(0, unbiased=False).clamp_min(0.05)
        return (pooled - mean) / scale, {"mean": mean, "scale": scale}

    def geometry_scores(support, features):
        return torch.cat([support.score(chunk)[0] for chunk in features.split(4096)]).numpy()

    raw, state = standardize(pool_nodes(values, outgoing, incoming, len(unique)))
    cached_features = {"raw": raw} if args.save_features else None
    artifact["raw_scaler"] = state
    raw_reference = raw[reference[:10000]]
    scores["raw_knn"] = knn_scores(raw_reference, raw).numpy()
    artifact["raw_reference"] = raw_reference
    for modes in (1, 8):
        support = ConditionalMahalanobis.fit(raw[reference], max_components=modes)
        scores[f"raw_mahalanobis_m{modes}"] = geometry_scores(support, raw)
        artifact["node_models"][f"raw_m{modes}"] = support.state_dict()
    deep = {1: [], 8: []}
    for seed, (model, flow_state) in enumerate(
        zip(detector.models, detector.state["seed_models"], strict=True)
    ):
        with torch.no_grad():
            embedded = (model.layers[:-1](values) - flow_state["embedding_mean"]) / flow_state[
                "embedding_scale"
            ]
        pooled, state = standardize(pool_nodes(embedded, outgoing, incoming, len(unique)))
        if cached_features is not None:
            cached_features[f"deep_{seed}"] = pooled
        for modes in (1, 8):
            support = ConditionalMahalanobis.fit(pooled[reference], max_components=modes)
            energy = geometry_scores(support, pooled)
            center, upper = np.quantile(energy[selection], [0.5, 0.95])
            scale = max(float(upper - center), 1e-6)
            deep[modes].append((energy - center) / scale + 1)
            artifact["node_models"][f"seed_{seed}_m{modes}"] = {
                **state,
                "support": support.state_dict(),
                "center": float(center),
                "scale": scale,
            }
    for modes in (1, 8):
        name = f"deep_mahalanobis_m{modes}_ensemble"
        scores[name] = np.mean(deep[modes], axis=0)
        raw_scale = max(float(np.quantile(scores["raw_knn"][selection], 0.95)), 1e-6)
        deep_scale = max(float(np.quantile(scores[name][selection], 0.95)), 1e-6)
        scores[f"raw_plus_deep_m{modes}"] = np.maximum(
            scores["raw_knn"] / raw_scale, scores[name] / deep_scale
        )
        artifact["node_models"][f"fusion_m{modes}"] = {"raw": raw_scale, "deep": deep_scale}
    torch.save(artifact, output / "node_detector.pt")
    node_order = np.argsort(node_minutes, kind="stable")
    boundaries = np.flatnonzero(np.r_[True, np.diff(node_minutes[node_order]) != 0])
    node_groups = {
        int(node_minutes[node_order[start]]): node_order[start:stop]
        for start, stop in zip(boundaries, np.r_[boundaries[1:], len(node_order)], strict=True)
    }
    queries = [minute for minute in sorted(truth) if cutoff // 60 <= minute < end // 60]
    if cached_features is not None:
        cache_path = Path(args.save_features).resolve()
        if cache_path.exists():
            raise ValueError("Refusing to overwrite an existing node feature cache")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "data_sha256": sha256(path),
                "dataset": dataset,
                "encoder": detector.state,
                "node_features": cached_features,
                "node_minutes": torch.from_numpy(node_minutes.copy()),
                # Equality keys for optional personalized memory; never neural inputs.
                "node_keys": torch.from_numpy(unique["node"].copy().view(np.int64)),
                "node_window_labels": torch.tensor([truth[int(t)]["label"] for t in node_minutes]),
                "source_reference": torch.from_numpy(reference),
                "source_selection": torch.from_numpy(selection),
                "source_calibration": torch.from_numpy(node_roles["calibration"]),
                "source_calibration_minutes": torch.tensor(sorted(role_windows["calibration"])),
                "target_minutes": torch.tensor(queries),
                "target_labels": torch.tensor([truth[t]["label"] for t in queries]),
                "target_types": [truth[t]["types"] for t in queries],
                "target_record_counts": torch.tensor([truth[t]["records"] for t in queries]),
                "source_role_windows": role_windows,
                "source_scalers": artifact,
            },
            cache_path,
        )
    archive = {}
    for name, energy in scores.items():
        windows = {
            minute: float(np.sort(energy[group])[-3:].mean())
            for minute, group in node_groups.items()
        }
        archive[name] = {
            "energy": torch.tensor(
                [windows.get(minute, 0.0) for minute in queries], dtype=torch.float64
            ),
            "observed": torch.tensor([minute in windows for minute in queries]),
            "source_calibration": torch.tensor(
                [windows[minute] for minute in sorted(role_windows["calibration"])],
                dtype=torch.float64,
            ),
            "source_calibration_minutes": torch.tensor(sorted(role_windows["calibration"])),
        }
    archive["record_count"] = {
        "energy": torch.tensor([np.log1p(truth[minute]["records"]) for minute in queries]),
        "observed": torch.ones(len(queries), dtype=torch.bool),
        "source_calibration": torch.tensor(
            [np.log1p(truth[minute]["records"]) for minute in sorted(role_windows["calibration"])]
        ),
    }
    torch.save(
        {
            "methods": archive,
            "minutes": torch.tensor(queries),
            "labels": torch.tensor([truth[minute]["label"] for minute in queries]),
            "types": [truth[minute]["types"] for minute in queries],
        },
        output / "scores.pt",
    )
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
            "unit": "directed_node_support_whole_minute",
            "source_normal_windows": {name: len(windows) for name, windows in role_windows.items()},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--keys", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-reference", type=int, default=20000)
    parser.add_argument("--buffer", type=int, default=20)
    parser.add_argument(
        "--save-features",
        help="Persist normal-source-scaled node vectors for reference-adaptation experiments",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
