"""Normal-only H2/H3 mechanism diagnostics, separate from network benchmarks.

All generators and budgets are fixed before examining these target outcomes.
These synthetic tests diagnose mechanisms; they do not prove site generality.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import fisher_exact, hypergeom
from sklearn.metrics import roc_auc_score

from .experiments import sha256, write_json
from .geometry import ConditionalMahalanobis
from .h1_classical_features import SourcePCA
from .h1_experiment import replay, support_values, units
from .h2_experiment import HEADS, KINDS, embeddings, make_model, model_input, validation_bce
from .h3_models import METHODS, SequentialMap, fit_gru, predict_gru, ridge_fit, ridge_predict

DOMAINS = (
    {"dimension": 8, "modes": 1, "noise": 0.05, "missing": 0.0},
    {"dimension": 12, "modes": 2, "noise": 0.15, "missing": 0.05},
    {"dimension": 16, "modes": 2, "noise": 0.30, "missing": 0.15},
)
POLICIES = ("fixed", "delayed_10pct", "immediate")
RIDGE_MODES = (1, 8)
HISTORY = 8
PURGE = 13


def encoded(values, present):
    return np.c_[values * present, present.astype(float)].astype(np.float64)


def row_draw(n, config, seed, family="normal"):
    """Parity sign triples and an independent correlated continuous pair.

    A parity flip preserves every univariate distribution and all population
    pair moments, including cross moments with the independent remaining block.
    Missingness is independent and copied unchanged for this construction.
    """
    rng = np.random.default_rng(seed)
    d = config["dimension"]
    regime = (
        rng.binomial(1, 0.8 if family == "drift" else 0.5, n)
        if config["modes"] == 2
        else np.zeros(n, int)
    )
    x = rng.normal(size=(n, d))
    signs = rng.choice((-1.0, 1.0), size=(n, 2))
    triple = np.c_[signs, signs[:, 0] * signs[:, 1]]
    x[:, :3] = triple * (1 + config["noise"] * np.abs(rng.normal(size=(n, 3))))
    u = rng.normal(
        loc=1.0 if family == "drift" else 0.0, scale=1.3 if family == "drift" else 1.0, size=n
    )
    x[:, 3] = u + regime
    x[:, 4] = 0.8 * u + regime + config["noise"] * rng.normal(size=n)
    if family == "parity_flip":
        x[:, 2] *= -1
    elif family == "relationship_reverse":
        x[:, 4] = -0.8 * u + regime + config["noise"] * rng.normal(size=n)
    elif family == "drift":
        x[:, :3] *= 1.3
        x[:, 5:-1] = 1.4 * x[:, 5:-1] + 1.5
    elif family != "normal":
        raise ValueError("Unknown row family")
    # The regime is an observed context coordinate, available to every model.
    x[:, -1] = regime
    present = rng.random((n, d)) >= config["missing"]
    present[:, -1] = True
    return encoded(x, present)


def generic_corrupt(raw, policy, seed):
    """Numeric/availability pairs are provenance groups before fixed PCA.

    Original: independently permute pairs with probability1/2. Conditional:
    observable regime and availability matched; singleton fallback retains
    regime and the copied group's availability. No latent oracle is used.
    """
    x = torch.as_tensor(raw, dtype=torch.float64)
    d = x.shape[1] // 2
    if x.shape[1] != 2 * d or not torch.isfinite(x).all():
        raise ValueError("Expected numeric values plus availability pairs")
    rng = np.random.default_rng(seed)
    result = x.clone()
    exact = np.c_[x[:, d - 1].numpy(), x[:, d:].numpy()]
    _, cohort = np.unique(exact, axis=0, return_inverse=True)
    for j in range(d):
        donor = rng.permutation(len(x))
        if policy == "conditional":
            for c in np.unique(cohort):
                ids = np.flatnonzero(cohort == c)
                pool = ids
                if len(ids) == 1:
                    pool = np.flatnonzero(
                        (x[:, d - 1].numpy() == x[ids[0], d - 1].item())
                        & (x[:, d + j].numpy() == x[ids[0], d + j].item())
                    )
                donor[ids] = rng.choice(pool, size=len(ids), replace=len(pool) < len(ids))
        elif policy != "original":
            raise ValueError("Unknown generic corruption")
        chosen = np.flatnonzero(rng.random(len(x)) < 0.5)
        result[chosen[:, None], torch.tensor([j, d + j])[None, :]] = x[donor[chosen]][:, [j, d + j]]
    return result


def temporal_draw(n, config, seed, drift=False):
    """Nonreversible isotropic rotation AR plus independent nuisance dimensions."""
    rng = np.random.default_rng(seed)
    d = config["dimension"]
    angle, radius = 0.55, 0.92
    a = radius * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    latent = np.empty((n, d))
    latent[0] = rng.normal(size=d)
    for i in range(1, n):
        latent[i, :2] = a @ latent[i - 1, :2] + math.sqrt(1 - radius**2) * rng.normal(size=2)
        latent[i, 2:] = 0.4 * latent[i - 1, 2:] + math.sqrt(0.84) * rng.normal(size=d - 2)
    regime = np.zeros(n, int)
    if config["modes"] == 2:
        for i in range(1, n):
            regime[i] = 1 - regime[i - 1] if rng.random() < 0.03 else regime[i - 1]
    latent[:, 2:-1] += regime[:, None]
    latent += config["noise"] * rng.normal(size=latent.shape)
    if drift:
        latent[:, :-1] = 1.3 * latent[:, :-1] + 1.5
    latent[:, -1] = regime
    present = rng.random(latent.shape) >= config["missing"]
    present[:, -1] = True
    return encoded(latent, present), a


def reverse_blocks(x, start, block=16):
    """Reverse each suffix block; preserve its exact multiset and missing masks."""
    result = x.copy()
    for i in range(start, len(x), block):
        result[i : i + block] = x[i : i + block][::-1]
    return result


def temporal_roles(counts):
    """13-minute endpoint purge makes all9 observed slots disjoint by role."""
    roles, cursor = {}, HISTORY
    for role, count in zip(("train", "selection", "support", "calibration"), counts, strict=True):
        roles[role] = np.arange(cursor, cursor + count)
        cursor += count + PURGE
    return roles, cursor


def role_observations(roles, history=0):
    owned = {name: np.unique(ids[:, None] - np.arange(history + 1)) for name, ids in roles.items()}
    names = list(owned)
    for i, name in enumerate(names):
        for other in names[i + 1 :]:
            if np.intersect1d(owned[name], owned[other]).size:
                raise ValueError("Synthetic source observation roles overlap")
    return owned


def sequence_examples(raw, adapter, query):
    z = adapter.transform(raw).numpy()
    query = np.asarray(query, dtype=np.int64)
    x, mask = np.zeros((len(query), HISTORY, 8)), np.zeros((len(query), HISTORY))
    for i, t in enumerate(query):
        for j, past in enumerate(range(int(t) - HISTORY, int(t))):
            if past >= 0:
                x[i, j], mask[i, j] = z[past], 1
    return {"x": x, "mask": mask, "y": z[query]}


def finite_metrics(flags, energy, labels, negative_control=False, policy="fixed"):
    labels = np.asarray(labels)
    attack, normal = labels == 1, labels == 0
    tp, fp = int(flags[attack].sum()), int(flags[normal].sum())
    result = {
        "tp": tp,
        "fp": fp,
        "attack": int(attack.sum()),
        "normal": int(normal.sum()),
        "attack_windows": int(attack.sum()),
        "normal_windows": int(normal.sum()),
        "tpr": float(flags[attack].mean()) if attack.any() else None,
        "fpr": float(flags[normal].mean()) if normal.any() else None,
        "auroc": float(roc_auc_score(labels, energy)) if attack.any() and normal.any() else None,
        "pauc_005": float(roc_auc_score(labels, energy, max_fpr=0.05))
        if attack.any() and normal.any()
        else None,
        "coverage": 1.0,
    }
    if negative_control:
        result["population_null"] = (
            "labels independent of the whole generated observation stream and history"
        )
        if attack.any() and normal.any() and policy == "fixed":
            table = [[tp, int(attack.sum()) - tp], [fp, int(normal.sum()) - fp]]
            result["fixed_policy_fisher_exact_two_sided_p"] = float(fisher_exact(table).pvalue)
            extremes = hypergeom.ppf(
                [0.025, 0.975], len(labels), int(flags.sum()), int(attack.sum())
            )
            result["fixed_policy_null95_tpr_minus_fpr"] = (
                extremes / attack.sum() - (flags.sum() - extremes) / normal.sum()
            ).tolist()
            result["formal_null_scope"] = (
                "fixed frozen flags and independent randomized labels; conditional exact test, "
                "unadjusted multiple comparisons"
            )
        else:
            result["formal_null_scope"] = (
                "adaptive feedback depends on past labels; descriptive rates only"
            )
        result["tpr_minus_fpr"] = (
            result["tpr"] - result["fpr"] if attack.any() and normal.any() else None
        )
    return result


def task_archive(output, name, methods, calibration, labels, negative_control=False):
    folder = output / name
    folder.mkdir()
    n = len(labels)
    minutes = np.arange(100000, 100000 + n)
    archive, decisions, report = {}, {}, {"task": name, "methods": {}}
    for method, energy in methods.items():
        source = calibration[method]
        archive[method] = {
            "energy": torch.as_tensor(energy),
            "source_calibration": torch.as_tensor(source),
        }
        report["methods"][method] = {}
        for policy in POLICIES:
            flags, thresholds = replay(
                np.asarray(energy), np.asarray(source), minutes, labels, policy
            )
            report["methods"][method][policy] = finite_metrics(
                flags, energy, labels, negative_control, policy
            )
            decisions[method + "/" + policy] = {
                "flags": torch.from_numpy(flags),
                "thresholds": torch.from_numpy(thresholds),
            }
    torch.save(
        {
            "methods": archive,
            "minutes": torch.from_numpy(minutes),
            "labels": torch.as_tensor(labels),
            "source_calibration_minutes": torch.arange(
                90000, 90000 + len(next(iter(calibration.values())))
            ),
        },
        folder / "scores.pt",
    )
    torch.save(decisions, folder / "decisions.pt")
    report["status"] = "completed"
    write_json(folder / "report.json", report)
    return report


def train_generic(model, kind, adapter, train, select, policy, seed, epochs):
    positive = model_input(adapter, train, kind)
    validation = model_input(adapter, select, kind)
    negative_validation = model_input(adapter, generic_corrupt(select, policy, 7189), kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    best = validation_bce(model, validation, negative_validation)
    weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history, wait = [{"epoch": 0, "selection_bce": best}], 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        negative = model_input(adapter, generic_corrupt(train, policy, seed * 1000 + epoch), kind)
        for ids in torch.randperm(len(train), generator=generator).split(256):
            logits = model(torch.cat((positive[ids], negative[ids])))
            y = torch.cat((torch.zeros(len(ids)), torch.ones(len(ids)))).double()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite generic training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        value = validation_bce(model, validation, negative_validation)
        history.append({"epoch": epoch, "selection_bce": value})
        if value < best:
            best, wait = value, 0
            weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if wait >= 5:
            break
    model.load_state_dict(weights)
    model.eval()
    return history


@torch.no_grad()
def row_scores(state, raw):
    adapter = SourcePCA.from_state_dict(state["adapter"])
    model = make_model(state["kind"], len(adapter.state["mean"]), state["seed"])
    model.load_state_dict(state["weights"])
    model.eval()
    feature = embeddings(model, model_input(adapter, raw, state["kind"]))
    support = ConditionalMahalanobis.from_state_dict(state["geometry"])
    values = {
        "scalar": model.readout(feature).squeeze(-1),
        "support": support_values(support, (feature - state["mean"]) / state["scale"]),
    }
    return {
        h: ((v - state["units"][h]["center"]) / state["units"][h]["scale"] + 1).numpy()
        for h, v in values.items()
    }


def row_ensemble_scores(state, raw):
    scored = [row_scores(seed, raw) for seed in state["seeds"]]
    result = {}
    for head in HEADS:
        u = state["ensemble_units"][head]
        result[head] = (np.mean([v[head] for v in scored], axis=0) - u["center"]) / u["scale"] + 1
    return result


@torch.no_grad()
def fit_row_state(model, kind, adapter, train, selection, seed, history):
    """Exactly the real H2 head rule: train features, SD1e-6, selection units."""
    rf = embeddings(model, model_input(adapter, train, kind))
    sf = embeddings(model, model_input(adapter, selection, kind))
    mean, scale = rf.mean(0), rf.std(0, unbiased=False).clamp_min(1e-6)
    support = ConditionalMahalanobis.fit((rf - mean) / scale, max_components=8)
    selected = {
        "scalar": model.readout(sf).squeeze(-1),
        "support": support_values(support, (sf - mean) / scale),
    }
    return {
        "adapter": adapter.state_dict(),
        "kind": kind,
        "seed": seed,
        "weights": model.state_dict(),
        "mean": mean,
        "scale": scale,
        "geometry": support.state_dict(),
        "units": {h: units(selected[h]) for h in HEADS},
        "history": history,
        "support_fit_role": "train",
        "support_fit_rows": len(train),
        "feature_sd_floor": 1e-6,
    }


def row_tasks(config, n):
    # Separate seeds are fixed by role/task, never selected after performance.
    normal = row_draw(n // 2, config, 8001)
    tasks = {
        "parity_flip": (
            np.r_[normal, row_draw(n - n // 2, config, 8002, "parity_flip")],
            np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)],
        ),
        "relationship_reverse": (
            np.r_[normal, row_draw(n - n // 2, config, 8003, "relationship_reverse")],
            np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)],
        ),
        "normal_drift": (row_draw(n, config, 8004, "drift"), np.zeros(n, int)),
        "identical_labels": (
            row_draw(n, config, 8005),
            np.random.default_rng(8006).integers(0, 2, n),
        ),
    }
    return tasks


def h2_domain(output, config, smoke):
    counts = (64, 32, 32, 32, 64) if smoke else (4096, 1024, 1024, 256, 1024)
    train, select, diagnostic_reference, cal = [
        torch.from_numpy(row_draw(n, config, 1001 + i)) for i, n in enumerate(counts[:4])
    ]
    adapter = SourcePCA.fit(train)
    boundaries = np.cumsum((0, *counts[:4]))
    role_ids = {
        name: np.arange(boundaries[i], boundaries[i + 1])
        for i, name in enumerate(("train", "selection", "diagnostic_reference", "calibration"))
    }
    source_ids = role_observations(role_ids)
    tasks = row_tasks(config, counts[4])
    raw_query = torch.from_numpy(np.concatenate([v[0] for v in tasks.values()]))
    methods = {task: {} for task in tasks}
    calibration = {}
    model_records = {}
    ensemble_audits = {}
    seeds = (17,) if smoke else (17, 29, 41)
    for policy in ("original", "conditional"):
        for kind in KINDS:
            ensembles, cal_ensemble = {h: [] for h in HEADS}, {h: [] for h in HEADS}
            selection_ensemble = {h: [] for h in HEADS}
            seed_states = []
            for seed in seeds:
                model = make_model(kind, train.shape[1], seed)
                history = train_generic(
                    model, kind, adapter, train, select, policy, seed, 1 if smoke else 30
                )
                state = fit_row_state(model, kind, adapter, train, select, seed, history)
                path = output / f"{policy}_{kind}_{seed}.pt"
                expected = row_scores(state, raw_query[:8])
                torch.save(state, path)
                restored = torch.load(path, weights_only=True)
                actual = row_scores(restored, raw_query[:8])
                if any(not np.array_equal(expected[h], actual[h]) for h in HEADS):
                    raise ValueError("Mechanism row model disk restore differs")
                scores = row_scores(restored, raw_query)
                c = row_scores(restored, cal)
                s = row_scores(restored, select)
                diagnostic = row_scores(restored, diagnostic_reference)
                for h in HEADS:
                    ensembles[h].append(scores[h])
                    cal_ensemble[h].append(c[h])
                    selection_ensemble[h].append(s[h])
                seed_states.append(restored)
                model_records[f"{policy}/{kind}/{seed}"] = {
                    "history": history,
                    "path": str(path),
                    "sha256": sha256(path),
                    "restore_exact": True,
                    "diagnostic_reference_median_score": {
                        h: float(np.median(diagnostic[h])) for h in HEADS
                    },
                }
                print("H2 mechanism", config["dimension"], policy, kind, seed, flush=True)
            ensemble_units = {}
            for h in HEADS:
                name = f"{policy}_{kind}_{h}"
                u = units(torch.from_numpy(np.mean(selection_ensemble[h], axis=0)))
                ensemble_units[h] = u
                combined = np.mean(ensembles[h], axis=0)
                combined = (combined - u["center"]) / u["scale"] + 1
                calibration[name] = (np.mean(cal_ensemble[h], axis=0) - u["center"]) / u[
                    "scale"
                ] + 1
                for index, task in enumerate(tasks):
                    methods[task][name] = combined[index * counts[4] : (index + 1) * counts[4]]
            artifact = {
                "seeds": seed_states,
                "ensemble_units": ensemble_units,
                "kind": kind,
                "corruption": policy,
            }
            ensemble_path = output / f"{policy}_{kind}_ensemble.pt"
            torch.save(artifact, ensemble_path)
            checked = row_ensemble_scores(
                torch.load(ensemble_path, weights_only=True), raw_query[:8]
            )
            first_task = next(iter(tasks))
            errors = {}
            for h in HEADS:
                expected = methods[first_task][f"{policy}_{kind}_{h}"][:8]
                errors[h] = float(np.max(np.abs(checked[h] - expected) / (1 + np.abs(expected))))
            if max(errors.values()) > 1e-7:
                raise ValueError("Mechanism ensemble disk restore differs")
            ensemble_audits[f"{policy}/{kind}"] = {
                "scaled_errors": errors,
                "path": str(ensemble_path),
                "sha256": sha256(ensemble_path),
            }
            write_json(output / f"progress_{policy}_{kind}.json", {"models": model_records})
    reports = {
        name: task_archive(
            output, name, methods[name], calibration, labels, name == "identical_labels"
        )
        for name, (_, labels) in tasks.items()
    }
    torch.save(
        {
            "train": train,
            "selection": select,
            "diagnostic_reference": diagnostic_reference,
            "calibration": cal,
            "observation_ids": {k: torch.from_numpy(v) for k, v in source_ids.items()},
            "role_generator_seeds": {
                "train": 1001,
                "selection": 1002,
                "diagnostic_reference": 1003,
                "calibration": 1004,
            },
            "tasks": {
                k: {"features": torch.from_numpy(v[0]), "labels": torch.from_numpy(v[1])}
                for k, v in tasks.items()
            },
        },
        output / "draws.pt",
    )
    return {
        "models": model_records,
        "ensemble_audits": ensemble_audits,
        "tasks": reports,
        "counts": counts,
        "observation_roles_disjoint": True,
    }


def temporal_representation(state, examples):
    x, mask, y = examples["x"], examples["mask"], examples["y"]
    if state["kind"] == "delay_support":
        return torch.from_numpy(np.c_[y, x.reshape(len(x), -1), mask])
    if state["kind"] == "gru":
        predicted = predict_gru(state["predictor"], x, mask)
    else:
        mapping = SequentialMap(state["kind"], state["seed"], state["map"])
        predicted = ridge_predict(state["predictor"], mapping.transform(x, mask))
    return torch.from_numpy(y - predicted)


def temporal_scores(state, examples):
    f = (temporal_representation(state, examples) - state["mean"]) / state["scale"]
    result = {}
    for mode in RIDGE_MODES:
        spec = state["supports"][str(mode)]
        score = support_values(ConditionalMahalanobis.from_state_dict(spec["geometry"]), f)
        result[mode] = ((score - spec["center"]) / spec["scale"] + 1).numpy()
    return result


def temporal_tasks(config, n):
    raw, matrix = temporal_draw(n + HISTORY, config, 9001)
    normal, _ = temporal_draw(n + HISTORY, config, 9002)
    drift, _ = temporal_draw(n + HISTORY, config, 9003, drift=True)
    labels = np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)]
    return {
        "order_reverse": (reverse_blocks(raw, HISTORY + n // 2), labels),
        "normal_drift": (drift, np.zeros(n, int)),
        "identical_labels": (normal, np.random.default_rng(9004).integers(0, 2, n)),
    }, matrix


def h3_domain(output, config, smoke):
    counts = (64, 32, 32, 32) if smoke else (256, 128, 128, 128)
    role_ids, length = temporal_roles(counts)
    source_ids = role_observations(role_ids, HISTORY)
    source, matrix = temporal_draw(length, config, 2001)
    fit_ids = np.unique(role_ids["train"][:, None] - np.arange(HISTORY + 1))
    adapter = SourcePCA.fit(torch.from_numpy(source[fit_ids]))
    examples = {role: sequence_examples(source, adapter, ids) for role, ids in role_ids.items()}
    n = 64 if smoke else 1024
    tasks, _ = temporal_tasks(config, n)
    query = {
        task: sequence_examples(raw, adapter, np.arange(HISTORY, HISTORY + n))
        for task, (raw, _) in tasks.items()
    }
    methods = {task: {} for task in tasks}
    calibration, models = {}, {}
    for kind in METHODS:
        pooled = {task: {m: [] for m in RIDGE_MODES} for task in (*tasks, "calibration")}
        for seed in (17,) if smoke else (17, 29, 41):
            state = {"kind": kind, "seed": seed}
            train, select = examples["train"], examples["selection"]
            if kind == "gru":
                state["predictor"] = fit_gru(
                    train["x"],
                    train["mask"],
                    train["y"],
                    select["x"],
                    select["mask"],
                    select["y"],
                    seed,
                    2 if smoke else 100,
                )
            elif kind != "delay_support":
                mapping = SequentialMap(kind, seed)
                state["map"] = mapping.state_dict()
                state["predictor"] = ridge_fit(
                    mapping.transform(train["x"], train["mask"]),
                    train["y"],
                    mapping.transform(select["x"], select["mask"]),
                    select["y"],
                )
            rf, sf = (
                temporal_representation(state, examples["support"]),
                temporal_representation(state, select),
            )
            state["mean"], state["scale"] = rf.mean(0), rf.std(0, unbiased=False).clamp_min(0.05)
            rf, sf = (rf - state["mean"]) / state["scale"], (sf - state["mean"]) / state["scale"]
            state["supports"] = {}
            for mode in RIDGE_MODES:
                support = ConditionalMahalanobis.fit(rf, max_components=mode)
                state["supports"][str(mode)] = {
                    "geometry": support.state_dict(),
                    **units(support_values(support, sf)),
                }
            expected = temporal_scores(state, query["order_reverse"])
            path = output / f"{kind}_{seed}.pt"
            torch.save({"adapter": adapter.state_dict(), "model": state}, path)
            restored = torch.load(path, weights_only=True)["model"]
            actual = temporal_scores(restored, query["order_reverse"])
            if any(not np.array_equal(expected[m], actual[m]) for m in RIDGE_MODES):
                raise ValueError("Mechanism temporal model disk restore differs")
            for task, ex in {**query, "calibration": examples["calibration"]}.items():
                scored = temporal_scores(restored, ex)
                for mode in RIDGE_MODES:
                    pooled[task][mode].append(scored[mode])
            models[f"{kind}/{seed}"] = {
                "path": str(path),
                "sha256": sha256(path),
                "restore_exact": True,
            }
            print("H3 mechanism", config["dimension"], kind, seed, flush=True)
        for mode in RIDGE_MODES:
            name = f"{kind}_m{mode}"
            calibration[name] = np.mean(pooled["calibration"][mode], axis=0)
            for task in tasks:
                methods[task][name] = np.mean(pooled[task][mode], axis=0)
        write_json(output / f"progress_{kind}.json", {"models": models})
    reports = {
        name: task_archive(
            output, name, methods[name], calibration, labels, name == "identical_labels"
        )
        for name, (_, labels) in tasks.items()
    }
    torch.save(
        {
            "source": torch.from_numpy(source),
            "role_endpoints": {k: torch.from_numpy(v) for k, v in role_ids.items()},
            "observation_ids": {k: torch.from_numpy(v) for k, v in source_ids.items()},
            "tasks": {
                k: {"features": torch.from_numpy(v[0]), "labels": torch.from_numpy(v[1])}
                for k, v in tasks.items()
            },
            "adapter_fit_observations": torch.from_numpy(fit_ids),
            "rotation_transition": torch.from_numpy(matrix),
        },
        output / "draws.pt",
    )
    return {
        "models": models,
        "tasks": reports,
        "counts": counts,
        "observation_roles_disjoint": True,
    }


def run(args):
    torch.set_num_threads(args.workers)
    torch.set_num_interop_threads(1)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    registration = {
        "smoke": args.smoke,
        "domains": DOMAINS,
        "seeds": [17] if args.smoke else [17, 29, 41],
        "source_sha256": sha256(Path(__file__)),
        "workers": args.workers,
        "scope": "synthetic mechanism diagnostics, no independent real-site or QPU claim",
        "negative_control": (
            "fixed-policy exact randomization null primary; adaptive rates descriptive, "
            "no seed regeneration"
        ),
    }
    write_json(output / "registration.json", registration)
    result = {"status": "running", "smoke": args.smoke, "domains": {}, "tasks": {}}
    score_archive, decision_archive = {}, {}
    try:
        for config in DOMAINS:
            name = f"d{config['dimension']}_m{config['modes']}"
            folder = output / name
            folder.mkdir()
            h2, h3 = folder / "h2", folder / "h3"
            h2.mkdir()
            h3.mkdir()
            result["domains"][name] = {
                "config": config,
                "observed_input_dimension": 2 * config["dimension"],
                "h2": h2_domain(h2, config, args.smoke),
                "h3": h3_domain(h3, config, args.smoke),
            }
            write_json(output / f"progress_{name}.json", result["domains"][name])
            for hypothesis in ("h2", "h3"):
                for family, task in result["domains"][name][hypothesis]["tasks"].items():
                    task_id = f"{hypothesis}/{name}/{family}"
                    path = folder / hypothesis / family
                    metadata = {"hypothesis": hypothesis, "domain": name, "family": family}
                    result["tasks"][task_id] = {**metadata, **task, "run": str(path)}
                    score_archive[task_id] = {
                        **metadata,
                        **torch.load(path / "scores.pt", weights_only=True),
                    }
                    decision_archive[task_id] = torch.load(path / "decisions.pt", weights_only=True)
        torch.save({"tasks": score_archive}, output / "scores.pt")
        torch.save({"tasks": decision_archive}, output / "decisions.pt")
        result.update(status="completed", elapsed_seconds=time.perf_counter() - started)
        write_json(output / "report.json", result)
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
    print("MECHANISMS COMPLETE", result["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
