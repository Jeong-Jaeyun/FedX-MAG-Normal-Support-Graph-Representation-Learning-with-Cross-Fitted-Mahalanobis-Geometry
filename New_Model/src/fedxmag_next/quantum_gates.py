"""Label-free capability gates for representation blocks (quantum or classical).

Each gate isolates one condition that can stop a representation block from helping a
normal-support anomaly detector. No gate reads attack labels. Passing a gate is a
precondition for a detection experiment, not evidence of detection gain.

G2 magnitude response   -- does support energy keep growing with outlier radius?
G3 readability          -- does the measured feature set retain input sensitivity?
G4 memory capacity      -- can recent inputs be linearly recovered from a temporal map?
GD geometric difference -- Huang et al., Nat. Commun. 12, 2631 (2021), arXiv:2011.01938,
                           Eqs. (L7)-(L11): can any labelling favour the quantum kernel
                           over the best of a suite of classical kernels on this data?
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.linalg
import torch

from .geometry import ConditionalMahalanobis

Transform = Callable[[np.ndarray], np.ndarray]

# Huang et al. (L10) regularisation grid and (L11) Gaussian bandwidth multipliers.
LAMBDAS = (1e-5, 1e-4, 1e-3, 1e-2, 0.025, 0.05, 0.1)
GAMMA_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
MAX_TRAINING_ERROR = 0.045


def _array(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Finite [rows, columns] values required")
    return values


# --------------------------------------------------------------------------- G2


def magnitude_response(
    transform: Transform,
    dimension: int,
    radii=(1, 2, 3, 5, 10, 30, 100),
    reference: int = 4096,
    directions: int = 200,
    seed: int = 0,
    max_components: int = 1,
) -> dict:
    """Median support energy at radius k along random directions, over the normal q95.

    Normal data are N(0, I) in ``dimension`` coordinates, so the typical normal radius
    is about sqrt(dimension). A magnitude-preserving block should cross the normal q95
    soon after the typical shell and keep increasing.
    """
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=(reference, dimension))
    feats = _array(transform(normal))
    support = ConditionalMahalanobis.fit(torch.from_numpy(feats), max_components=max_components)
    q95 = float(np.quantile(support.score(torch.from_numpy(feats))[0].numpy(), 0.95))
    unit = rng.normal(size=(directions, dimension))
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    ratios = []
    for k in radii:
        energy = support.score(torch.from_numpy(_array(transform(k * unit))))[0].numpy()
        ratios.append(float(np.median(energy) / q95))
    typical = float(np.sqrt(dimension))
    beyond = [r for k, r in zip(radii, ratios, strict=True) if k >= typical]
    detection = next((k for k, r in zip(radii, ratios, strict=True) if r >= 1.0), None)
    # Saturation: energy far outside must keep separating from energy just outside, so
    # rare normals and extreme anomalies do not tie. Quadratic growth gives ~100x here.
    anchor = next((r for k, r in zip(radii, ratios, strict=True) if k >= 2 * typical), None)
    growth = None if anchor is None else ratios[-1] / max(anchor, 1e-12)
    monotone = bool(np.all(np.diff(beyond) >= -1e-9))
    return {
        "radii": list(radii),
        "median_energy_over_normal_q95": ratios,
        "typical_radius": typical,
        "monotone_beyond_typical": monotone,
        "detection_radius": detection,
        "detection_radius_over_typical": None if detection is None else detection / typical,
        "growth_far_over_near": growth,
        "passes": bool(
            detection is not None
            and detection <= 3 * typical
            and monotone
            and growth is not None
            and growth >= 4.0
        ),
    }


# --------------------------------------------------------------------------- G3


def jacobian_readability(transform: Transform, points, eps: float = 1e-5) -> dict:
    """Central-difference input Jacobian of the feature map at normal points."""
    z = _array(points)
    n, d = z.shape
    columns = []
    for j in range(d):
        step = np.zeros(d)
        step[j] = eps
        columns.append((_array(transform(z + step)) - _array(transform(z - step))) / (2 * eps))
    jac = np.stack(columns, axis=2)  # [n, features, inputs]
    sv = np.linalg.svd(jac, compute_uv=False)
    participation = (sv**2).sum(1) ** 2 / np.maximum((sv**4).sum(1), 1e-300)
    return {
        "inputs": d,
        "features": int(jac.shape[1]),
        "frobenius_mean": float(np.linalg.norm(jac, axis=(1, 2)).mean()),
        "sigma_min_mean": float(sv[:, -1].mean()),
        "sigma_max_mean": float(sv[:, 0].mean()),
        "effective_rank_mean": float(participation.mean()),
        "rank_deficient_fraction": float((sv[:, -1] < 1e-6 * np.maximum(sv[:, 0], 1e-300)).mean()),
    }


def local_bloch_length(features, qubits: int) -> dict:
    """Mean single-qubit Bloch length for features laid out as X(q), Y(q), Z(q), ...

    For a pure global state, a length below one means that qubit's information has moved
    into multi-qubit correlations that single-qubit observables do not read.
    """
    f = _array(features)
    if f.shape[1] < 3 * qubits:
        raise ValueError("Features must start with X, Y and Z blocks")
    xyz = np.stack([f[:, :qubits], f[:, qubits : 2 * qubits], f[:, 2 * qubits : 3 * qubits]], 2)
    length = np.linalg.norm(xyz, axis=2)
    return {
        "mean": float(length.mean()),
        "q05": float(np.quantile(length, 0.05)),
        "q50": float(np.quantile(length, 0.50)),
        "feature_sd_median": float(np.median(f.std(0))),
    }


# --------------------------------------------------------------------------- G4


def memory_capacity(features_train, features_test, history_train, history_test,
                    ridges=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0), validation: float = 0.25) -> dict:
    """Linear recall R^2 of z_{t-k} (k=1 most recent) from features of an input history.

    ``history_*`` is [rows, steps, inputs]; features must be computed from that history
    only. Features are standardised on the fit rows; the ridge strength per lag is chosen
    on the last ``validation`` fraction of the training rows, then refit on all training
    rows. Negative test R^2 is clipped at zero before summing (Jaeger-style capacity).
    """
    ftr, fte = _array(features_train), _array(features_test)
    htr, hte = np.asarray(history_train, float), np.asarray(history_test, float)
    if htr.ndim != 3 or hte.shape[1:] != htr.shape[1:]:
        raise ValueError("History must be [rows, steps, inputs]")
    mean, scale = ftr.mean(0), ftr.std(0)
    scale[scale < 1e-12] = 1.0
    a = np.c_[(ftr - mean) / scale, np.ones(len(ftr))]
    b = np.c_[(fte - mean) / scale, np.ones(len(fte))]
    cut = int(len(a) * (1 - validation))
    eye = np.eye(a.shape[1])
    eye[-1, -1] = 0.0  # never penalise the intercept

    def solve(rows, y, lam):
        x = a[rows]
        return np.linalg.lstsq(x.T @ x / len(x) + lam * eye, x.T @ y / len(x), rcond=None)[0]

    def r2(pred, y):
        res = ((pred - y) ** 2).sum(0)
        tot = np.maximum(((y - y.mean(0)) ** 2).sum(0), 1e-300)
        return float(np.mean(1 - res / tot))

    steps = htr.shape[1]
    per_lag, chosen = [], []
    for lag in range(1, steps + 1):
        ytr, yte = htr[:, steps - lag], hte[:, steps - lag]
        fit, val = np.arange(cut), np.arange(cut, len(a))
        lam = max(ridges, key=lambda r: r2(a[val] @ solve(fit, ytr[fit], r), ytr[val]))
        w = solve(np.arange(len(a)), ytr, lam)
        per_lag.append(float(max(0.0, r2(b @ w, yte))))
        chosen.append(lam)
    return {"per_lag_r2": per_lag, "capacity": float(sum(per_lag)), "max": steps,
            "ridge_per_lag": chosen}


# --------------------------------------------------------------------------- GD


def trace_normalize(kernel) -> np.ndarray:
    k = _array(kernel)
    if k.shape[0] != k.shape[1]:
        raise ValueError("Square kernel matrix required")
    k = (k + k.T) / 2
    return k * (len(k) / np.trace(k))


def linear_kernel(x) -> np.ndarray:
    x = _array(x)
    return x @ x.T


def rbf_kernel(x, gamma: float) -> np.ndarray:
    x = _array(x)
    sq = (x**2).sum(1)
    dist = np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0)
    return np.exp(-gamma * dist)


def projected_quantum_kernel(features, qubits: int, gamma: float) -> np.ndarray:
    """Huang et al. (L6): exp(-gamma * sum_k sum_P (Tr(P rho_k) - Tr(P rho'_k))^2)."""
    f = _array(features)[:, : 3 * qubits]
    return rbf_kernel(f, gamma)


def fidelity_kernel(statevectors) -> np.ndarray:
    s = np.asarray(statevectors)
    if s.ndim != 2:
        raise ValueError("Statevectors must be [rows, amplitudes]")
    return np.abs(s.conj() @ s.T) ** 2


def median_gamma(values) -> float:
    """Label-free bandwidth: inverse median squared pairwise distance."""
    x = _array(values)
    sq = (x**2).sum(1)
    dist = np.maximum(sq[:, None] + sq[None, :] - 2 * x @ x.T, 0.0)
    upper = dist[np.triu_indices(len(x), 1)]
    return 1.0 / max(float(np.median(upper)), 1e-12)


def _largest_eigenvalue(m: np.ndarray) -> float:
    m = (m + m.T) / 2
    n = len(m)
    return float(scipy.linalg.eigh(m, eigvals_only=True, subset_by_index=[n - 1, n - 1])[0])


def _psd_sqrt(k: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(k)
    return (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T


def approximate_dimension(kernel) -> float:
    """Huang et al. (L7) with the 1/(N-k+1) normalisation that makes it span [1, N]."""
    t = np.clip(np.linalg.eigvalsh(trace_normalize(kernel))[::-1], 0.0, None)
    n = len(t)
    tails = np.cumsum(t[::-1])[::-1]  # tails[k-1] = sum_{l>=k} t_l
    return float((tails / (n - np.arange(n))).sum())


def geometric_difference(k_quantum, k_classical, lambdas=LAMBDAS,
                         max_training_error: float = MAX_TRAINING_ERROR) -> dict:
    """g_gen(K_C || K_Q) at the largest lambda whose g_tra stays below the bound (L8-L9)."""
    kq, kc = trace_normalize(k_quantum), trace_normalize(k_classical)
    if kq.shape != kc.shape:
        raise ValueError("Kernels must cover the same rows")
    sq = _psd_sqrt(kq)
    w, v = np.linalg.eigh(kc)
    w = np.clip(w, 0.0, None)
    rows = []
    a = sq @ v
    for lam in lambdas:
        gen = (a * (w / (w + lam) ** 2)) @ a.T
        tra = (a * (1.0 / (w + lam) ** 2)) @ a.T
        rows.append({
            "lambda": float(lam),
            "g_gen": float(np.sqrt(max(_largest_eigenvalue(gen), 0.0))),
            "g_tra": float(lam * np.sqrt(max(_largest_eigenvalue(tra), 0.0))),
        })
    valid = [r for r in rows if r["g_tra"] < max_training_error]
    chosen = max(valid, key=lambda r: r["lambda"]) if valid else None
    n = len(kq)
    return {
        "n": n,
        "sqrt_n": float(np.sqrt(n)),
        "chosen": chosen,
        "g": None if chosen is None else chosen["g_gen"],
        "g_over_sqrt_n": None if chosen is None else chosen["g_gen"] / np.sqrt(n),
        "grid": rows,
    }


def minimum_classical_geometric_difference(k_quantum, x_classical) -> dict:
    """Minimum g over the linear kernel and the (L11) Gaussian bandwidth grid on x."""
    x = _array(x_classical)
    base = 1.0 / (x.shape[1] * max(float(x.var()), 1e-12))
    candidates = {"linear": linear_kernel(x)}
    for m in GAMMA_MULTIPLIERS:
        candidates[f"rbf_{m:g}"] = rbf_kernel(x, m * base)
    results = {name: geometric_difference(k_quantum, k) for name, k in candidates.items()}
    finite = {name: r["g"] for name, r in results.items() if r["g"] is not None}
    best = min(finite, key=finite.get) if finite else None
    return {
        "minimum_g": None if best is None else finite[best],
        "minimum_g_over_sqrt_n": None if best is None else results[best]["g_over_sqrt_n"],
        "closest_classical_kernel": best,
        "by_kernel": {name: r["g"] for name, r in results.items()},
    }
