"""Positive control (synthetic) and real normal-relation predictability (label-free).

The real-data part reads only rows from the saved H1 source-adapter normal fit indices.
Targets are whitened normal principal coordinates 9..(8+targets) predicted from the
8-d PCA input that the quantum blocks receive. No attack labels or target windows.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fedxmag_next.experiments import sha256
from fedxmag_next.quantum_positive_control import positive_control, relation_predictability

DATASETS = {
    "cic": "New_Model/artifacts/cic_all_legacy_http_v1/records.npz",
    "xiiotid": "New_Model/artifacts/xiiotid_context_all_v2/records.npz",
    "ton": "New_Model/artifacts/ton_protocol_context_v1/records.npz",
}


def real_split(root, dataset, targets, seed):
    adapter = torch.load(root / f"New_Model/runs/round233_h1_{dataset}_pilot_v2/source_adapter.pt",
                         weights_only=True)
    state = adapter["adapter"]
    fit = np.sort(adapter["fit_indices"].numpy())
    path = root / DATASETS[dataset]
    with np.load(path) as data:
        if np.any(data["label"][fit]):
            raise ValueError("Source fit rows must be normal")
        x = (data["features"][fit].astype(np.float64) - state["mean"].numpy()) / state[
            "scale"
        ].numpy()
    lam, vec = np.linalg.eigh(x.T @ x / len(x))
    lam, vec = lam[::-1], vec[:, ::-1]
    keep = np.flatnonzero(lam > 1e-8 * lam[0])
    if len(keep) < 8 + targets:
        raise ValueError("Too few nondegenerate normal directions")
    proj = x @ vec[:, keep[: 8 + targets]] / np.sqrt(lam[keep[: 8 + targets]])
    z, y = proj[:, :8], proj[:, 8:]
    order = np.random.default_rng(seed).permutation(len(z))
    return z[order], y[order], {"data_sha256": sha256(path), "rows": len(z),
                                "nondegenerate": int(len(keep))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"D:\FedX_MAG")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--targets", type=int, default=16)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "claim": "positive control is engineered (Huang et al. App. G); real part is "
                 "label-free normal relation predictability; exact CPU simulation",
        "args": vars(args),
        "implementation_sha256": {
            p.name: sha256(p) for p in (
                Path(__file__),
                Path(__file__).parents[1] / "src/fedxmag_next/quantum_positive_control.py",
                Path(__file__).parents[1] / "src/fedxmag_next/quantum_gates.py",
            )
        },
        "positive_control": {},
        "real_relations": {},
    }
    for kernel in ("quantum_fidelity", "quantum_projected_g1"):
        runs = []
        for seed in args.seeds:
            r = positive_control(kernel, n_anchor=400, n_train=1000, n_val=500, n_cal=500,
                                 n_test=1000, noise=0.1, seed=seed)
            runs.append(r)
            print("positive", kernel, seed,
                  {f: round(v["auroc"], 3) for f, v in r["families"].items()}, flush=True)
        report["positive_control"][kernel] = runs
    for dataset in DATASETS:
        z, y, prov = real_split(Path(args.root), dataset, args.targets, 17)
        tr, va, te = slice(0, 1500), slice(1500, 2000), slice(2000, 3000)
        r = relation_predictability(z[tr], y[tr], z[va], y[va], z[te], y[te])
        r["provenance"] = prov
        report["real_relations"][dataset] = r
        print("real", dataset,
              {f: round(v["r2_vs_mean"], 4) for f, v in r["families"].items()}, flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("done", round(report["elapsed_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
