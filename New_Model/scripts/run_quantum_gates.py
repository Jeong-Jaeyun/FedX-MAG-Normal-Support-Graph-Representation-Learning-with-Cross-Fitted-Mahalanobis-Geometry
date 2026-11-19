"""Label-free capability-gate baseline for existing H1/H3 representation blocks.

Never reads attack labels or target windows. Real-data rows are drawn only from the
saved H1 source-adapter normal fit indices. Results are gate measurements, not
detection performance and not a quantum-advantage claim.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fedxmag_next.experiments import sha256
from fedxmag_next.h1_classical_features import ClassicalFeatureMap, SourcePCA
from fedxmag_next.h3_models import SequentialMap
from fedxmag_next.quantum_features import ProjectedQuantumMap
from fedxmag_next.quantum_gates import (
    approximate_dimension,
    fidelity_kernel,
    geometric_difference,
    jacobian_readability,
    local_bloch_length,
    magnitude_response,
    median_gamma,
    memory_capacity,
    minimum_classical_geometric_difference,
    projected_quantum_kernel,
)

DATASETS = {
    "cic": "New_Model/artifacts/cic_all_legacy_http_v1/records.npz",
    "xiiotid": "New_Model/artifacts/xiiotid_context_all_v2/records.npz",
    "ton": "New_Model/artifacts/ton_protocol_context_v1/records.npz",
}
CLASSICAL = ("linear", "trigonometric", "rff", "random_mlp", "rbf_nystrom", "polynomial_nystrom")


def block_maps(reference):
    maps = {"raw": lambda z: np.asarray(z, float)}
    for kind in CLASSICAL:
        fitted = ClassicalFeatureMap.fit(kind, torch.from_numpy(reference), seed=17)
        maps[kind] = lambda z, m=fitted: m.transform(torch.from_numpy(np.asarray(z))).numpy()
    for kind, ent in (("quantum", True), ("quantum_noent", False)):
        circuit = ProjectedQuantumMap(seed=17, entangled=ent)
        maps[kind] = lambda z, c=circuit: c.transform(np.asarray(z)).numpy()
    return maps


def synthetic_gates(seed):
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=(4096, 8))
    points = rng.normal(size=(256, 8))
    out = {}
    for name, transform in block_maps(reference).items():
        row = {
            "magnitude": magnitude_response(transform, 8, seed=seed),
            "readability": jacobian_readability(transform, points),
        }
        if name.startswith("quantum"):
            row["bloch"] = local_bloch_length(transform(points), 8)
        out[name] = row
        print("synthetic", name, "G2", row["magnitude"]["passes"], flush=True)
    return out


def memory_gates(seed):
    rng = np.random.default_rng(seed)
    h_tr, h_te = rng.normal(size=(2000, 8, 8)), rng.normal(size=(1000, 8, 8))
    m_tr, m_te = np.ones((2000, 8)), np.ones((1000, 8))
    out = {}
    for kind in ("ar_ridge", "esn", "quantum", "quantum_noent", "quantum_zero_history"):
        sm = SequentialMap(kind, seed=17)
        out[kind] = memory_capacity(sm.transform(h_tr, m_tr), sm.transform(h_te, m_te), h_tr, h_te)
        print("memory", kind, round(out[kind]["capacity"], 3), flush=True)
    return out


def kernel_gates(z, label):
    """Geometric difference of quantum kernels against classical kernels on the same z."""
    kernels = {}
    for name, ent in (("quantum", True), ("quantum_noent", False)):
        circuit = ProjectedQuantumMap(seed=17, entangled=ent)
        kernels[f"{name}_fidelity"] = fidelity_kernel(circuit.statevectors(z))
        xyz = circuit.transform(z).numpy()[:, :24]
        gamma = median_gamma(xyz)
        for mult in (0.25, 1.0, 4.0):
            kernels[f"{name}_projected_g{mult:g}"] = projected_quantum_kernel(xyz, 8, mult * gamma)
    out = {"rows": len(z), "kernels": {}}
    for name, k in kernels.items():
        row = minimum_classical_geometric_difference(k, z)
        row["approximate_dimension"] = approximate_dimension(k)
        out["kernels"][name] = row
        print(label, name, "g_min", row["minimum_g"], "dim", round(row["approximate_dimension"], 1),
              flush=True)
    # Does entanglement add geometry that the product circuit cannot express?
    out["entangled_vs_product"] = {
        kind: geometric_difference(
            kernels[f"quantum_{kind}"], kernels[f"quantum_noent_{kind}"]
        )["g"]
        for kind in ("fidelity", "projected_g1")
    }
    return out


def real_rows(root, dataset, rows, seed):
    adapter = torch.load(root / f"New_Model/runs/round233_h1_{dataset}_pilot_v2/source_adapter.pt",
                         weights_only=True)
    fit = adapter["fit_indices"].numpy()
    pick = np.sort(np.random.default_rng(seed).choice(fit, rows, replace=False))
    path = root / DATASETS[dataset]
    with np.load(path) as data:
        if np.any(data["label"][pick]):
            raise ValueError("Source fit rows must be normal")
        features = torch.from_numpy(data["features"][pick].astype(np.float64))
    z = SourcePCA.from_state_dict(adapter["adapter"]).transform(features).numpy()
    return z, {"data_sha256": sha256(path), "rows": rows, "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"D:\FedX_MAG")
    parser.add_argument("--output", required=True)
    parser.add_argument("--kernel-rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {
        "claim": "label-free gate measurements; no attack labels; exact CPU simulation",
        "args": vars(args),
        "implementation_sha256": {
            p.name: sha256(p)
            for p in (Path(__file__),
                      Path(__file__).parents[1] / "src/fedxmag_next/quantum_gates.py")
        },
        "synthetic": synthetic_gates(args.seed),
        "memory": memory_gates(args.seed),
        "kernels": {},
    }
    synthetic_z = np.random.default_rng(args.seed).normal(size=(args.kernel_rows, 8))
    report["kernels"]["synthetic_gaussian"] = kernel_gates(synthetic_z, "synthetic")
    for dataset in DATASETS:
        z, provenance = real_rows(Path(args.root), dataset, args.kernel_rows, 17)
        report["kernels"][dataset] = {"provenance": provenance, **kernel_gates(z, dataset)}
    report["elapsed_seconds"] = time.perf_counter() - started
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print("done", round(report["elapsed_seconds"], 1), flush=True)


if __name__ == "__main__":
    main()
