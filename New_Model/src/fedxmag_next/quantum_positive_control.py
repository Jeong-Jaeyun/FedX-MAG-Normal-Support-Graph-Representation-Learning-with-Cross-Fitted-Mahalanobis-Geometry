"""Quantum-favourable positive control and normal-relation predictability (label-free).

Positive control: following Huang et al. (2021) Appendix G, build a function f that is
simple in a quantum kernel's RKHS and complex in the classical kernel's RKHS. Normal rows
are (x, t = f(x) + noise); anomalies swap t between rows, so marginals are preserved and
only the relation breaks. A normal-only kernel-ridge regressor x -> t is scored by its
squared residual. If the quantum kernel cannot win here under the same normal-only
protocol, the protocol/head is the limiting condition rather than the data.

Relation predictability: on real normal rows only, predict held-out coordinates from the
8-d block input with quantum or classical kernels. Hyper-parameters are selected by
normal validation MSE only. No attack labels are read.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import roc_auc_score

from .quantum_features import ProjectedQuantumMap
from .quantum_gates import (
    GAMMA_MULTIPLIERS,
    _psd_sqrt,
    median_gamma,
    trace_normalize,
)

RIDGES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
KernelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _sqdist(a, b):
    return np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T, 0.0)


def rbf(gamma: float) -> KernelFn:
    return lambda a, b: np.exp(-gamma * _sqdist(a, b))


def linear() -> KernelFn:
    return lambda a, b: a @ b.T


def quantum_kernels(seed: int = 17, gamma_reference=None) -> dict[str, KernelFn]:
    """Fidelity and projected kernels of the existing H1 circuit, entangled and product."""
    kernels = {}
    for name, ent in (("quantum", True), ("quantum_noent", False)):
        circuit = ProjectedQuantumMap(seed=seed, entangled=ent)

        def fidelity(a, b, c=circuit):
            return np.abs(c.statevectors(a).conj() @ c.statevectors(b).T) ** 2

        kernels[f"{name}_fidelity"] = fidelity
        if gamma_reference is not None:
            xyz_ref = circuit.transform(gamma_reference).numpy()[:, :24]
            base = median_gamma(xyz_ref)
            for mult in (0.25, 1.0, 4.0):
                def projected(a, b, c=circuit, g=mult * base):
                    fa = c.transform(a).numpy()[:, :24]
                    fb = c.transform(b).numpy()[:, :24]
                    return np.exp(-g * _sqdist(fa, fb))

                kernels[f"{name}_projected_g{mult:g}"] = projected
    return kernels


def classical_kernels(reference) -> dict[str, KernelFn]:
    x = np.asarray(reference, float)
    base = 1.0 / (x.shape[1] * max(float(x.var()), 1e-12))
    out = {"linear": linear()}
    for m in GAMMA_MULTIPLIERS:
        out[f"rbf_{m:g}"] = rbf(m * base)
    return out


def fit_kernel_ridge(kernel: KernelFn, x_train, y_train, x_val, y_val, ridges=RIDGES) -> dict:
    """Select the ridge strength by normal validation MSE; never refit on validation rows."""
    k = kernel(x_train, x_train)
    k = (k + k.T) / 2
    kv = kernel(x_val, x_train)
    w, v = np.linalg.eigh(k)
    w = np.clip(w, 0.0, None)
    center = y_train.mean(0)
    proj = v.T @ (y_train - center)
    best = None
    n = len(k)
    for lam in ridges:
        alpha = v @ (proj / (w + lam * n)[:, None])
        mse = float(((kv @ alpha + center - y_val) ** 2).mean())
        if best is None or mse < best["validation_mse"]:
            best = {"ridge": lam, "alpha": alpha, "center": center, "validation_mse": mse}
    best["x_train"] = x_train
    best["kernel"] = kernel
    return best


def predict(model: dict, x) -> np.ndarray:
    return model["kernel"](x, model["x_train"]) @ model["alpha"] + model["center"]


def best_of(kernels: dict[str, KernelFn], x_train, y_train, x_val, y_val) -> dict:
    fits = {name: fit_kernel_ridge(k, x_train, y_train, x_val, y_val)
            for name, k in kernels.items()}
    name = min(fits, key=lambda n: fits[n]["validation_mse"])
    return {"name": name, "model": fits[name],
            "validation_mse": {n: f["validation_mse"] for n, f in fits.items()}}


# ------------------------------------------------------------------ positive control


def favourable_function(kq_fn: KernelFn, kc_fn: KernelFn, anchors, lam: float = 1e-3,
                        jitter: float = 1e-6):
    """f(x) = sum_i alpha_i k_Q(x, anchor_i) maximising s_C / s_Q (Huang et al., G3-G4)."""
    kq = trace_normalize(kq_fn(anchors, anchors))
    kc = trace_normalize(kc_fn(anchors, anchors))
    scale_q = len(anchors) / np.trace(kq_fn(anchors, anchors))
    sq = _psd_sqrt(kq)
    w, v = np.linalg.eigh(kc)
    w = np.clip(w, 0.0, None)
    a = sq @ v
    m = (a * (w / (w + lam) ** 2)) @ a.T
    evals, evecs = np.linalg.eigh((m + m.T) / 2)
    y = sq @ evecs[:, -1]
    alpha = np.linalg.solve(kq + jitter * np.eye(len(kq)), y)

    def f(x):
        return scale_q * kq_fn(x, anchors) @ alpha

    return f, {"g_gen_squared": float(evals[-1]), "anchors": len(anchors), "lambda": lam}


def relation_rows(f, n: int, dimension: int, noise: float, seed: int, scale=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, dimension))
    t = f(x)
    if scale is None:
        scale = (float(t.mean()), float(t.std()) or 1.0)
    t = (t - scale[0]) / scale[1] + noise * rng.normal(size=n)
    return x, t[:, None], scale


def positive_control(kq_name: str, n_anchor=400, n_train=1000, n_val=500, n_cal=500,
                     n_test=1000, noise=0.1, seed=0, dimension=8,
                     adversary: str = "classical") -> dict:
    """Run the relation-swap detection task; returns metrics per kernel family.

    ``adversary="classical"`` builds f against the closest linear/Gaussian kernel (v1).
    ``adversary="classical_plus_product"`` also admits the product-circuit kernels, which
    are efficiently classical, so f must need entanglement to stay simple.
    """
    if adversary not in ("classical", "classical_plus_product"):
        raise ValueError("Unknown adversary suite")
    rng = np.random.default_rng(seed)
    anchors = rng.normal(size=(n_anchor, dimension))
    qk = quantum_kernels(gamma_reference=anchors)
    ck = classical_kernels(anchors)
    from .quantum_gates import geometric_difference, minimum_classical_geometric_difference

    kq_anchor = qk[kq_name](anchors, anchors)
    gd = minimum_classical_geometric_difference(kq_anchor, anchors)
    reference = ck[gd["closest_classical_kernel"]]
    if adversary == "classical_plus_product":
        product = {k: v for k, v in qk.items() if k.startswith("quantum_noent")}
        product_g = {k: geometric_difference(kq_anchor, v(anchors, anchors))["g"]
                     for k, v in product.items()}
        product_g = {k: g for k, g in product_g.items() if g is not None}
        gd["product_circuit_g"] = product_g
        if product_g and min(product_g.values()) < gd["minimum_g"]:
            name = min(product_g, key=product_g.get)
            gd["minimum_g"], gd["closest_classical_kernel"] = product_g[name], name
            reference = product[name]
        ck = {**ck, **product}
    gd["adversary"] = adversary
    f, meta = favourable_function(qk[kq_name], reference, anchors)
    x_tr, t_tr, scale = relation_rows(f, n_train, dimension, noise, seed + 1)
    x_va, t_va, _ = relation_rows(f, n_val, dimension, noise, seed + 2, scale)
    x_ca, t_ca, _ = relation_rows(f, n_cal, dimension, noise, seed + 3, scale)
    x_te, t_te, _ = relation_rows(f, n_test, dimension, noise, seed + 4, scale)
    half = n_test // 2
    t_anom = t_te.copy()
    t_anom[half:] = np.random.default_rng(seed + 5).permutation(t_te[half:, 0])[:, None]
    labels = np.r_[np.zeros(half), np.ones(n_test - half)]
    families = {
        "quantum_target": {kq_name: qk[kq_name]},
        "quantum_all": {k: v for k, v in qk.items() if k.startswith("quantum_")
                        and not k.startswith("quantum_noent")},
        "product_circuit": {k: v for k, v in qk.items() if k.startswith("quantum_noent")},
        "classical_suite": ck,
    }
    out = {"kernel": kq_name, "geometric_difference": gd, "function": meta,
           "noise": noise, "rows": {"train": n_train, "val": n_val, "cal": n_cal,
                                    "test": n_test}, "families": {}}
    for fam, kernels in families.items():
        best = best_of(kernels, x_tr, t_tr, x_va, t_va)
        model = best["model"]
        cal = ((predict(model, x_ca) - t_ca) ** 2).ravel()
        score = ((predict(model, x_te) - t_anom) ** 2).ravel()
        threshold = float(np.quantile(cal, 0.95))
        flags = score > threshold
        out["families"][fam] = {
            "selected": best["name"],
            "validation_mse": best["validation_mse"][best["name"]],
            "test_normal_mse": float(((predict(model, x_te[:half]) - t_te[:half]) ** 2).mean()),
            "auroc": float(roc_auc_score(labels, score)),
            "tpr": float(flags[labels == 1].mean()),
            "fpr": float(flags[labels == 0].mean()),
        }
    return out


# ------------------------------------------------------------------ real normal relations


def relation_predictability(z_train, y_train, z_val, y_val, z_test, y_test) -> dict:
    """Held-out normal MSE predicting y (e.g. residual-subspace coordinates) from z."""
    qk = quantum_kernels(gamma_reference=z_train[:512])
    families = {
        "quantum": {k: v for k, v in qk.items() if not k.startswith("quantum_noent")},
        "product_circuit": {k: v for k, v in qk.items() if k.startswith("quantum_noent")},
        "classical_suite": classical_kernels(z_train),
        "mean_only": None,
    }
    var = float(((y_test - y_train.mean(0)) ** 2).mean())
    out = {"targets": int(y_train.shape[1]), "mean_only_mse": var, "families": {}}
    for fam, kernels in families.items():
        if kernels is None:
            continue
        best = best_of(kernels, z_train, y_train, z_val, y_val)
        mse = float(((predict(best["model"], z_test) - y_test) ** 2).mean())
        out["families"][fam] = {"selected": best["name"], "test_mse": mse,
                                "r2_vs_mean": 1 - mse / max(var, 1e-300)}
    return out
