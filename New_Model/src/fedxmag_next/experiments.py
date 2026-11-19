"""Measured development experiments on the local, reusable research datasets.

All fit roles are normal only. Attack labels are used for reporting only.
Every attempted method is retained, including failures. Results are development
evidence because the same datasets have informed earlier model choices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import Tensor

from .contracts import GraphWindow, clean_batch, collate, load_windows
from .detector import conformal_threshold
from .geometry import ConditionalMahalanobis
from .model import CausalEdgeGraphEncoder, EncoderConfig
from .training import TrainingConfig, train_encoder

ROLE_FILES = {
    "train": "train_normal.pt",
    "selection": "select_normal.pt",
    "calibration": "calibration_normal.pt",
    "test_normal": "test_normal.pt",
    "test_attack": "test_attack.pt",
    "audit_normal": "audit_normal.pt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    if path.name == "registration.json" and isinstance(value, dict):
        source_root = Path(__file__).resolve().parent
        snapshot = path.parent / "source_snapshot"
        snapshot.mkdir(parents=True, exist_ok=False)
        hashes = {}
        for source in sorted(source_root.glob("*.py")):
            data = source.read_bytes()
            (snapshot / source.name).write_bytes(data)
            hashes[source.name] = hashlib.sha256(data).hexdigest()
        value = {**value, "implementation_sha256": hashes}
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_roles(root: Path) -> tuple[dict[str, list[GraphWindow]], dict[str, Path]]:
    prepared = root / "data/processed/xiiotid_stage1_v3b_development"
    paths = {role: prepared / name for role, name in ROLE_FILES.items()}
    paths["audit_attack"] = root / "data/processed/xiiotid_stage1_v3b_audit_band/audit_attack.pt"
    if paths["test_attack"].stat().st_size == 0:
        from .recovery import recover

        paths["test_attack"] = recover(root)
    windows = {role: load_windows(path) for role, path in paths.items()}
    seen = set()
    for role, items in windows.items():
        if role in {"train", "selection", "calibration"} and any(w.label != 0 for w in items):
            raise ValueError(f"Attack in fit role {role}")
        for item in items:
            key = (item.metadata["session_key"], item.timestamp())
            if key in seen:
                raise ValueError(f"Duplicate window across roles: {key}")
            seen.add(key)
    return windows, paths


@torch.no_grad()
def encode(
    model: CausalEdgeGraphEncoder, windows: Sequence[GraphWindow], batch_size: int = 64
) -> dict[str, list[Tensor]]:
    model.eval()
    result: dict[str, list[Tensor]] = {"node": [], "edge": [], "node_error": [], "edge_error": []}
    for start in range(0, len(windows), batch_size):
        batch = collate(windows[start : start + batch_size])
        output = model(clean_batch(batch))
        for kind in ("node", "edge"):
            index = getattr(batch, f"{kind}_graph")
            counts = torch.bincount(index, minlength=batch.size).tolist()
            state = getattr(output, f"{kind}_embedding").detach().cpu()
            prediction = getattr(output, f"{kind}_prediction")
            target = getattr(batch, f"{kind}_features")
            available = getattr(batch, f"{kind}_available")
            error = ((prediction - target) * available).square().sum(dim=1)
            error = error / available.sum(dim=1).clamp_min(1)
            result[kind].extend(state.split(counts))
            result[f"{kind}_error"].extend(error.split(counts))
    return result


def raw_rows(windows: Sequence[GraphWindow], kind: str) -> list[Tensor]:
    return [
        torch.cat(
            (
                getattr(w, f"{kind}_features") * getattr(w, f"{kind}_available"),
                getattr(w, f"{kind}_available").float(),
            ),
            dim=1,
        ).double()
        for w in windows
    ]


def aggregate_rows(values: Tensor, rows: Sequence[Tensor], tail: int = 3) -> np.ndarray:
    parts = values.split([len(row) for row in rows])
    return np.asarray(
        [
            float(part.topk(min(tail, len(part))).values.mean()) if len(part) else 0.0
            for part in parts
        ]
    )


def knn_scores(
    reference: Tensor,
    query: Tensor,
    k: int = 20,
    chunk: int = 512,
    *,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    # Squared-distance cancellation in float32 can overwhelm the distance
    # between near-duplicate telemetry rows. Keep the geometry in float64.
    reference, query = reference.to(dtype), query.to(dtype)
    if not len(reference):
        raise ValueError("kNN normal reference cannot be empty")
    if not len(query):
        return torch.empty(0, dtype=torch.float64, device=query.device)
    results = []
    for begin in range(0, len(query), chunk):
        distances = torch.cdist(query[begin : begin + chunk], reference)
        results.append(distances.topk(min(k, len(reference)), largest=False).values.mean(dim=1))
    return torch.cat(results).double()


def channel_scores(
    rows: dict[str, list[Tensor]], algorithm: str, *, components: int = 8, seed: int = 17
) -> dict[str, np.ndarray]:
    reference = torch.cat(rows["train"]).double()
    # Train-only standardization. Constant dimensions are dropped so availability
    # flags which never vary cannot create artificial inverse-covariance axes.
    scale = reference.std(dim=0, unbiased=False)
    keep = scale > 1e-6
    mean, scale = reference.mean(dim=0)[keep], scale[keep]
    transformed = {
        role: (torch.cat(value).double()[:, keep] - mean) / scale for role, value in rows.items()
    }
    if algorithm == "knn":
        entity = {
            role: knn_scores(transformed["train"], value)
            for role, value in transformed.items()
            if role != "train"
        }
    else:
        support = ConditionalMahalanobis.fit(
            transformed["train"], max_components=components, min_component_size=32, seed=seed
        )
        entity = {
            role: support.score(value)[0] for role, value in transformed.items() if role != "train"
        }
    return {role: aggregate_rows(value, rows[role]) for role, value in entity.items()}


def fuse(channels: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    scales = [max(float(np.quantile(channel["selection"], 0.95)), 1e-9) for channel in channels]
    return {
        role: np.max(
            np.stack(
                [channel[role] / scale for channel, scale in zip(channels, scales, strict=True)]
            ),
            axis=0,
        )
        for role in channels[0]
    }


def metrics(calibration: np.ndarray, normal: np.ndarray, attack: np.ndarray) -> dict:
    threshold = conformal_threshold(calibration, 0.05)
    if threshold is None:
        raise ValueError("Insufficient normal calibration for the registered 5% target")
    labels = np.concatenate((np.zeros(len(normal)), np.ones(len(attack))))
    values = np.concatenate((normal, attack))
    fpr, tpr, _ = roc_curve(labels, values)
    return {
        "auroc": float(roc_auc_score(labels, values)),
        "auprc": float(average_precision_score(labels, values)),
        "threshold": threshold,
        "tpr": float(np.mean(attack > threshold)),
        "fpr": float(np.mean(normal > threshold)),
        "tpr_at_eval_fpr_005_diagnostic_only": float(tpr[fpr <= 0.05].max()),
        "normal_count": len(normal),
        "attack_count": len(attack),
    }


def summarize(scores: dict[str, np.ndarray], windows: dict[str, list[GraphWindow]]) -> dict:
    report = {}
    for band in ("test", "audit"):
        normal_key, attack_key = f"{band}_normal", f"{band}_attack"
        result = metrics(scores["calibration"], scores[normal_key], scores[attack_key])
        threshold = result["threshold"]
        sessions = np.asarray([w.metadata["session_key"] for w in windows[normal_key]])
        result["per_capture_fpr"] = {
            capture: {
                "normal_count": int(np.sum(sessions == capture)),
                "fpr": float(np.mean(scores[normal_key][sessions == capture] > threshold)),
            }
            for capture in sorted(set(sessions))
        }
        supported = [
            entry["fpr"]
            for entry in result["per_capture_fpr"].values()
            if entry["normal_count"] >= 10
        ]
        result["worst_capture_fpr_min10"] = max(supported) if supported else None
        report[band] = result
    return report


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    root, destination = Path(args.root).resolve(), Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    windows, paths = load_roles(root)
    cache_path = root / "output/root_cause_20261026/flow_context_cache.pt"
    cache = torch.load(cache_path, weights_only=True)
    for role, path in paths.items():
        if cache["bindings"][role] != sha256(path):
            if role != "test_attack" or path.parent.name != "recovered_v3b":
                raise ValueError(f"Cached flow alignment differs for {role}")
            diagnostic = json.loads((root / "output/diagnosis/test_band_windows.json").read_text())
            archived_keys = {
                (w["session"], w["window_start_ns"]) for w in diagnostic if w["attack_records"] > 0
            }
            derived_keys = {(w.metadata["session_key"], w.timestamp()) for w in windows[role]}
            if derived_keys != archived_keys:
                raise ValueError("Recovered attack windows differ from archived diagnostic keys")
        if len(cache["features"][role]) != len(windows[role]):
            raise ValueError(f"Cached flow count differs for {role}")
    registration = {
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "iterative_development_on_reused_datasets",
        "args": vars(args),
        "inputs": {role: sha256(path) for role, path in paths.items()},
        "flow_cache_sha256": sha256(cache_path),
        "selection_policy": "fixed epochs; no attack-label model selection",
        "success_gate": (
            "Must improve over raw-node+flow kNN on both bands, then survive "
            "capture-disjoint and second-dataset tests; a smoke test is not success."
        ),
        "roles": {role: len(items) for role, items in windows.items()},
    }
    write_json(destination / "registration.json", registration)
    print(json.dumps({"phase": "registered", "counts": registration["roles"]}), flush=True)
    started = time.perf_counter()
    channels = {}
    raw = {
        kind: {role: raw_rows(items, kind) for role, items in windows.items()}
        for kind in ("node", "edge")
    }
    for kind in ("node", "flow"):
        rows = raw["node"] if kind == "node" else cache["features"]
        channels[f"raw_{kind}_knn"] = channel_scores(rows, "knn")
    methods = {"raw_node_flow_knn": fuse([channels["raw_node_knn"], channels["raw_flow_knn"]])}
    torch.manual_seed(args.seed)
    encoder = CausalEdgeGraphEncoder(EncoderConfig(14, 7, args.hidden_dim, args.layers))
    history = train_encoder(
        encoder,
        windows["train"],
        TrainingConfig(epochs=args.epochs, batch_size=32, seed=args.seed),
        verbose=True,
    )
    torch.save(
        {"config": encoder.config.to_dict(), "model": encoder.state_dict(), "history": history},
        destination / "encoder.pt",
    )
    encoded = {role: encode(encoder, items) for role, items in windows.items()}
    torch.save(encoded, destination / "embeddings.pt")
    for components in (1, 8):
        neural = []
        skip = []
        for kind in ("node", "edge"):
            neural_rows = {role: values[kind] for role, values in encoded.items()}
            neural.append(channel_scores(neural_rows, "mahalanobis", components=components))
            skip_rows = {
                role: [
                    torch.cat((a.double(), b.double()), dim=1)
                    for a, b in zip(raw[kind][role], values[kind], strict=True)
                ]
                for role, values in encoded.items()
            }
            skip.append(channel_scores(skip_rows, "mahalanobis", components=components))
        methods[f"gnn_mahalanobis_k{components}"] = fuse(neural)
        methods[f"raw_skip_gnn_mahalanobis_k{components}"] = fuse(skip)
    learned_knn = []
    errors = []
    for kind in ("node", "edge"):
        learned_knn.append(
            channel_scores({role: values[kind] for role, values in encoded.items()}, "knn")
        )
        errors.append(
            {
                role: aggregate_rows(torch.cat(values[f"{kind}_error"]), values[kind])
                for role, values in encoded.items()
                if role != "train"
            }
        )
    methods["gnn_knn"] = fuse(learned_knn)
    methods["gnn_reconstruction_error"] = fuse(errors)
    methods["raw_flow_plus_gnn_knn"] = fuse(
        [*learned_knn, channels["raw_node_knn"], channels["raw_flow_knn"]]
    )
    methods["raw_flow_plus_gnn_error"] = fuse(
        [*errors, channels["raw_node_knn"], channels["raw_flow_knn"]]
    )
    report = {"status": registration["status"], "methods": {}}
    for name, score in methods.items():
        report["methods"][name] = summarize(score, windows)
        print(
            json.dumps(
                {
                    "method": name,
                    "metrics": {
                        band: {key: result[key] for key in ("auroc", "tpr", "fpr")}
                        for band, result in report["methods"][name].items()
                    },
                }
            ),
            flush=True,
        )
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(destination / "report.json", report)
    torch.save(
        {
            name: {role: torch.from_numpy(value) for role, value in score.items()}
            for name, score in methods.items()
        },
        destination / "scores.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
