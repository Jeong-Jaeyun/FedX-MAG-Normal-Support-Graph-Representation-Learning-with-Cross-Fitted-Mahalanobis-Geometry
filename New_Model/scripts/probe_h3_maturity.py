"""Normal-source-only H3 memory/readout diagnostic; never evaluates attacks."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from fedxmag_next.experiments import sha256, write_json
from fedxmag_next.h1_classical_features import SourcePCA
from fedxmag_next.h3_data import histories, summarize
from fedxmag_next.h3_models import SequentialMap

LAMBDAS = (1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e6)


def regress(train, valid, y, vy, strengths=LAMBDAS):
    x, sx, target, validation = [torch.as_tensor(v).double() for v in (train, valid, y, vy)]
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(0.05)
    xx, ss = (x - mean) / scale, (sx - mean) / scale
    center = target.mean(0)
    gram, right = xx.T @ xx / len(xx), xx.T @ (target - center) / len(xx)
    candidates = []
    for strength in strengths:
        weight = torch.linalg.solve(
            gram + strength * torch.eye(gram.shape[0], dtype=torch.float64), right
        )
        mse = float((ss @ weight + center - validation).square().mean())
        if not np.isfinite(mse):
            raise ValueError("Nonfinite normal-source decoder loss")
        candidates.append({"ridge": strength, "validation_mse": mse})
    baseline = float((validation - center).square().mean())
    return {
        "selected": min(candidates, key=lambda c: c["validation_mse"]),
        "constant_mean_mse": baseline,
        "candidates": candidates,
    }


def nonlinear(train, valid, y, vy):
    """Same 128-landmark RBF decoder search budget for every representation."""
    x, sx = torch.as_tensor(train).double(), torch.as_tensor(valid).double()
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(0.05)
    x, sx = (x - mean) / scale, (sx - mean) / scale
    median = float(torch.pdist(x[:512]).square().median().clamp_min(1e-6))
    ids = np.random.default_rng(731).permutation(len(x))[:128]
    landmarks = x[ids]
    candidates = []
    for factor in (0.1, 1.0, 10.0):
        gamma = factor / median
        kernel = torch.exp(-gamma * torch.cdist(landmarks, landmarks).square())
        eigen, axes = torch.linalg.eigh(kernel)
        inverse_root = axes / eigen.clamp_min(max(float(eigen.max()) * 1e-8, 1e-10)).sqrt()
        tx = torch.exp(-gamma * torch.cdist(x, landmarks).square()) @ inverse_root
        vx = torch.exp(-gamma * torch.cdist(sx, landmarks).square()) @ inverse_root
        result = regress(tx, vx, y, vy, (1e-8, 1e-4, 1e-2, 1.0, 100.0))
        candidates += [{"gamma_factor": factor, "gamma": gamma, **c} for c in result["candidates"]]
    return {
        "landmarks": len(ids),
        "landmark_seed": 731,
        "selected": min(candidates, key=lambda c: c["validation_mse"]),
        "candidates": candidates,
    }


def load_source(root, dataset):
    import json

    run = root / f"New_Model/runs/round235_h3_{dataset}_v1"
    reg = json.loads((run / "registration.json").read_text(encoding="utf-8"))
    source = torch.load(run / "source_adapter.pt", weights_only=True)
    ids = np.union1d(
        source["observation_ids"]["train"].numpy(), source["observation_ids"]["selection"].numpy()
    )
    if np.intersect1d(
        source["observation_ids"]["train"], source["observation_ids"]["selection"]
    ).size:
        raise ValueError("Overlapping source roles")
    data = Path(reg["input_audit"]["data"])
    if sha256(data) != reg["input_audit"]["data_sha256"]:
        raise ValueError("Prepared data changed")
    with np.load(data) as payload:
        features = payload["features"][ids]
        times = payload["observed_at"][ids]
    if times.max() // 60 >= reg["input_audit"]["cutoff_minute"]:
        raise ValueError("Source probe reached target chronology")
    minutes, summaries = summarize(features, times)
    adapter = SourcePCA.from_state_dict(source["adapter"])
    z = adapter.transform(summaries).numpy()
    examples = {}
    for role in ("train", "selection"):
        query = source["example_minutes"][role].numpy()
        if not np.isin(query, minutes).all():
            raise ValueError("Incomplete source query")
        x, mask = histories(query, minutes, z, np.ones(len(z), bool))
        if not (mask == 1).all():
            raise ValueError("Incomplete source history")
        examples[role] = {"x": x, "mask": mask, "y": z[np.searchsorted(minutes, query)]}
    return (
        run,
        examples,
        {
            "data_sha256": reg["input_audit"]["data_sha256"],
            "source_adapter_sha256": sha256(run / "source_adapter.pt"),
            "train_examples": len(examples["train"]["y"]),
            "selection_examples": len(examples["selection"]["y"]),
            "source_rows": len(ids),
            "max_observed_at": int(times.max()),
            "target_cutoff_minute": reg["input_audit"]["cutoff_minute"],
            "roles_disjoint": True,
            "target_labels_or_scores_used": False,
        },
    )


def run(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = {
        "status": "running",
        "script_sha256": sha256(Path(__file__)),
        "source_only": True,
        "seed": 17,
        "target_evaluation": False,
        "purpose": "readout and information-retention diagnostics, not attack-model selection",
        "datasets": {},
    }
    for dataset in ("cic", "xiiotid", "ton"):
        original, ex, provenance = load_source(root, dataset)
        train, select = ex["train"], ex["selection"]
        entry = {"source": provenance, "methods": {}}
        for kind in ("quantum", "quantum_noent", "esn", "ar_ridge"):
            saved = torch.load(original / f"{kind}_detector.pt", weights_only=True)
            seed = next(s for s in saved["seeds"] if s["seed"] == 17)
            mapping = SequentialMap(kind, 17, seed["map"])
            f = mapping.transform(train["x"], train["mask"])
            sf = mapping.transform(select["x"], select["mask"])
            reversed_f = mapping.transform(train["x"][:, ::-1], train["mask"][:, ::-1])
            removed_x, removed_mask = train["x"].copy(), train["mask"].copy()
            removed_x[:, -1], removed_mask[:, -1] = 0, 0
            removed_f = mapping.transform(removed_x, removed_mask)
            sd = f.std(0)
            stats = {
                "dimensions": f.shape[1],
                "feature_sd_median": float(np.median(sd)),
                "nonmask_sd_median": float(np.median(sd[:-8])),
                "sd_below_floor005": int((sd < 0.05).sum()),
                "reversal_feature_rms": float(np.mean((f - reversed_f) ** 2) ** 0.5),
                "last_slot_removal_rms": float(np.mean((f - removed_f) ** 2) ** 0.5),
                "stored_next_z_selection_mse": seed["predictor"]["validation_mse"],
                "stored_ridge": seed["predictor"].get("ridge"),
                "model_sha256": sha256(original / f"{kind}_detector.pt"),
            }
            for name, y, vy in (
                ("next_z", train["y"], select["y"]),
                ("last_z", train["x"][:, -1], select["x"][:, -1]),
            ):
                stats[name] = {"expanded_linear": regress(f, sf, y, vy)}
                if kind != "ar_ridge":
                    stats[name]["rbf_nystrom128"] = nonlinear(f, sf, y, vy)
            entry["methods"][kind] = stats
            print(dataset, kind, "source-only decoder checked", flush=True)
        result["datasets"][dataset] = entry
        write_json(out / f"{dataset}.json", entry)
    result.update(
        status="completed",
        elapsed_seconds=time.perf_counter() - started,
        qualification=(
            "same source selection reused across diagnostic choices; not independent generalization"
        ),
    )
    write_json(out / "report.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
