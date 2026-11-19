"""Preregistered full-window H1 static-representation pilot; CPU simulation."""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .detector import conformal_threshold
from .experiments import sha256, write_json
from .geometry import ConditionalMahalanobis
from .h1_classical_features import ClassicalFeatureMap, SourcePCA
from .quantum_features import ProjectedQuantumMap

PACKETS = {"cic": "cic_buffer39", "ton": "ton_buffer39", "xiiotid": "xiiotid_evidence_buffer50"}
KINDS = (
    "raw",
    "linear",
    "trigonometric",
    "rff",
    "random_mlp",
    "rbf_nystrom",
    "polynomial_nystrom",
    "quantum_noent",
    "quantum",
)
MODES = (1, 8)


def replay(energy, source, minutes, labels, policy, capacity=50):
    """Only confirmed past normal scores enter history; no target-fitted threshold."""
    history = deque(source[-capacity:].tolist(), maxlen=capacity)
    pending = deque()
    chosen = np.random.default_rng(731).random(len(minutes)) < 0.1
    flags = np.zeros(len(minutes), bool)
    thresholds = np.empty(len(minutes))
    for i, minute in enumerate(minutes):
        now = (int(minute) + 1) * 60
        while pending and pending[0][0] <= now:
            history.append(pending.popleft()[1])
        threshold = conformal_threshold(history, 0.05)
        if threshold is None:
            raise ValueError("Insufficient source calibration")
        thresholds[i] = threshold
        flags[i] = energy[i] > threshold
        if labels[i] == 0:
            if policy == "immediate":
                history.append(float(energy[i]))
            elif policy == "delayed_10pct" and chosen[i]:
                pending.append((now + 300, float(energy[i])))
            elif policy not in ("fixed", "delayed_10pct"):
                raise ValueError("Unknown feedback policy")
    return flags, thresholds


def metrics(flags, energy, labels, types):
    normal, attack = labels == 0, labels == 1
    result = {
        "tp": int(flags[attack].sum()),
        "fp": int(flags[normal].sum()),
        "attack_windows": int(attack.sum()),
        "normal_windows": int(normal.sum()),
        "tpr": float(flags[attack].mean()),
        "fpr": float(flags[normal].mean()),
        "coverage": 1.0,
        "auroc_score_diagnostic": float(roc_auc_score(labels, energy)),
        "standardized_pauc_005_diagnostic": float(roc_auc_score(labels, energy, max_fpr=0.05)),
        "per_type": {},
    }
    for kind in sorted({k for group in types for k in group}):
        mask = np.array([kind in group for group in types]) & attack
        result["per_type"][kind] = {
            "windows": int(mask.sum()),
            "detected": int(flags[mask].sum()),
            "tpr": float(flags[mask].mean()),
        }
    return result


def support_values(support, x):
    return torch.cat([support.score(chunk)[0] for chunk in x.split(4096)])


def restore_map(kind, state):
    if kind == "raw":
        return None
    if kind.startswith("quantum"):
        return ProjectedQuantumMap.from_state_dict(state)
    return ClassicalFeatureMap.from_state_dict(state)


class H1Detector:
    """Frozen row scorer, independent of query labels or target state."""

    def __init__(self, state):
        if state["format_version"] != 1:
            raise ValueError("Unsupported H1 artifact")
        self.state = state
        self.adapter = SourcePCA.from_state_dict(state["adapter"])
        self.models = [(s, restore_map(state["kind"], s["map"])) for s in state["seeds"]]

    def score(self, features):
        values = (
            self.adapter.normalize(features)
            if self.state["kind"] == "raw"
            else self.adapter.transform(features)
        )
        scores = {m: [] for m in MODES}
        for seed, mapping in self.models:
            f = values if mapping is None else mapping.transform(values)
            f = (f - seed["mean"]) / seed["scale"]
            for m in MODES:
                g = seed["supports"][str(m)]
                e = support_values(ConditionalMahalanobis.from_state_dict(g["geometry"]), f)
                scores[m].append((e - g["center"]) / g["scale"] + 1)
        result = {}
        for m in MODES:
            units = self.state["ensemble_units"][str(m)]
            result[m] = (torch.stack(scores[m]).mean(0) - units["center"]) / units["scale"] + 1
        return result


def units(values):
    median, q95 = torch.quantile(values, torch.tensor([0.5, 0.95], dtype=torch.float64))
    return {"center": float(median), "scale": max(float(q95 - median), 1e-6)}


def run(args):
    torch.set_num_threads(args.workers)
    root = Path(args.root).resolve()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    packet_path = (
        root / "New_Model/candidates/prototype_structure_v3" / (PACKETS[args.dataset] + ".pt")
    )
    packet = torch.load(packet_path, weights_only=True)
    base = packet["prototype_model"]["base_encoder"]
    risk = root / packet["prototype_model"]["protocol"]["args"]["run"]
    reg = json.loads((risk / "registration.json").read_text())
    data_path = root / reg["args"]["data"]
    digest = sha256(data_path)
    if digest != packet["data_sha256"] or digest != base["source_data_sha256"]:
        raise ValueError("Changed source dataset")
    cache = torch.load(risk / "row_scores.pt", weights_only=True)
    if cache["data_sha256"] != digest:
        raise ValueError("Changed row cache data")
    if sha256(risk / "row_scores.pt") != packet["prototype_model"]["protocol"]["row_cache_sha256"]:
        raise ValueError("Changed row cache")
    census = torch.load(risk / "scores.pt", weights_only=True)
    minutes, labels, types = census["minutes"].numpy(), census["labels"].numpy(), census["types"]
    cal_minutes = packet["source_window_calibration_minutes"].numpy()
    with np.load(data_path) as data:
        features = torch.from_numpy(data["features"])
        ts, row_labels, signature = data["observed_at"], data["label"], data["signature"]
        feature_names = data["feature_names"].tolist()
    if tuple(feature_names) != tuple(base["feature_names"]):
        raise ValueError("Changed source schema")
    roles = {k: v.numpy() for k, v in base["source_role_indices"].items()}
    joined = np.concatenate(list(roles.values()))
    if (
        len(np.unique(joined)) != len(joined)
        or np.any(row_labels[joined])
        or ts[joined].max() >= minutes.min() * 60
    ):
        raise ValueError("Invalid source roles")
    fit = roles["train"][np.argsort(signature[roles["train"]], kind="stable")[: args.reference]]
    selection = roles["selection"]
    combined = cache["combined_indices"].numpy()
    expected = np.concatenate((roles["calibration"], np.flatnonzero(np.isin(ts // 60, minutes))))
    if not np.array_equal(np.sort(expected), np.sort(combined)):
        raise ValueError("Combined observations do not cover every original target row")
    if len(np.unique(combined)) != len(combined) or cal_minutes.max() >= minutes.min():
        raise ValueError("Duplicate rows or future calibration")
    if not np.isin(cal_minutes, ts[roles["calibration"]] // 60).all():
        raise ValueError("Calibration source mismatch")
    for minute in cal_minutes:
        ids = np.flatnonzero(ts // 60 == minute)
        if np.any(row_labels[ids]) or not np.isin(ids, roles["calibration"]).all():
            raise ValueError("Incomplete normal calibration window")
    registration = {
        "status": "running",
        "args": vars(args),
        "data_sha256": digest,
        "packet_sha256": sha256(packet_path),
        "row_cache_sha256": sha256(risk / "row_scores.pt"),
        "window_census_sha256": sha256(risk / "scores.pt"),
        "training_rows": len(fit),
        "selection_rows": len(selection),
        "scored_rows": len(combined),
        "target_windows": len(minutes),
        "source_calibration_windows": len(cal_minutes),
        "kinds": KINDS,
        "seeds": [17, 29, 41],
        "modes": MODES,
        "primary_modes": 8,
        "aggregation": "top3",
        "normal_memory": False,
        "temporal_evidence": False,
        "claim": "reused development data; exact classical simulation; no quantum advantage claim",
        "controls": "all new models same4096 source fit; frozen DNN20k separate stronger reference",
        "implementation_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    }
    write_json(out / "registration.json", registration)
    print(args.dataset, "inputs verified", registration["scored_rows"], flush=True)
    adapter = SourcePCA.fit(features[fit])
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "fit_indices": torch.from_numpy(fit),
            "source_role_indices": base["source_role_indices"],
            "combined_indices": torch.from_numpy(combined),
        },
        out / "source_adapter.pt",
    )
    zfit, zsel = adapter.transform(features[fit]), adapter.transform(features[selection])
    zquery = torch.empty((len(combined), 8), dtype=torch.float64)
    for start in range(0, len(combined), 8192):
        zquery[start : start + 8192] = adapter.transform(features[combined[start : start + 8192]])
    row_minutes = ts[combined] // 60
    order = np.argsort(row_minutes, kind="stable")
    distinct, counts = np.unique(row_minutes[order], return_counts=True)
    groups = dict(zip(distinct.tolist(), np.split(order, np.cumsum(counts)[:-1]), strict=True))
    if any(int(m) not in groups for m in minutes):
        raise ValueError("A target window has no observations")
    allminutes = np.concatenate((cal_minutes, minutes))
    score_archive, decisions, report = (
        {},
        {},
        {"dataset": args.dataset, "methods": {}, "audits": {}},
    )

    def record(name, rows):
        windows = np.array(
            [
                np.partition(rows[groups[int(m)]], max(0, len(groups[int(m)]) - 3))[-3:].mean()
                for m in allminutes
            ]
        )
        cal, energy = windows[: len(cal_minutes)], windows[len(cal_minutes) :]
        score_archive[name] = {
            "source_calibration": torch.from_numpy(cal),
            "energy": torch.from_numpy(energy),
        }
        report["methods"][name] = {}
        for policy in ("fixed", "delayed_10pct", "immediate"):
            flags, thresholds = replay(energy, cal, minutes, labels, policy)
            report["methods"][name][policy] = metrics(flags, energy, labels, types)
            decisions[name + "/" + policy] = {
                "flags": torch.from_numpy(flags),
                "thresholds": torch.from_numpy(thresholds),
            }
        m = report["methods"][name]["immediate"]
        print(name, "immediate", m["tp"], m["fp"], flush=True)
        write_json(out / f"progress_{len(report['methods']):03d}.json", report)

    # Established row scores recombined with the pilot's common top3/policies.
    for m in MODES:
        record(
            f"frozen_dnn20k_m{m}", cache["row_scores"][f"deep_mahalanobis_m{m}_ensemble"].numpy()
        )
    raw_rows = {}
    for kind in KINDS:
        tick = time.perf_counter()
        state = {
            "format_version": 1,
            "kind": kind,
            "adapter": adapter.state_dict(),
            "feature_names": feature_names,
            "seeds": [],
            "ensemble_units": {},
        }
        selected_sum = {m: torch.zeros(len(selection), dtype=torch.float64) for m in MODES}
        query_sum = {m: np.zeros(len(combined)) for m in MODES}
        seeds = [17] if kind in ("raw", "linear", "trigonometric") else [17, 29, 41]
        for seed in seeds:
            if kind.startswith("quantum"):
                mapping = ProjectedQuantumMap(seed=seed, entangled=kind == "quantum")
            elif kind == "raw":
                mapping = None
            else:
                mapping = ClassicalFeatureMap.fit(kind, zfit, seed=seed)
            train = adapter.normalize(features[fit]) if mapping is None else mapping.transform(zfit)
            selected = (
                adapter.normalize(features[selection])
                if mapping is None
                else mapping.transform(zsel)
            )
            fmean, fscale = (
                (
                    torch.zeros(train.shape[1], dtype=torch.float64),
                    torch.ones(train.shape[1], dtype=torch.float64),
                )
                if kind == "raw"
                else (train.mean(0), train.std(0, unbiased=False).clamp_min(1e-6))
            )
            train, selected = (train - fmean) / fscale, (selected - fmean) / fscale
            seed_state = {
                "seed": seed,
                "map": {} if mapping is None else mapping.state_dict(),
                "mean": fmean,
                "scale": fscale,
                "supports": {},
            }
            geometries = {}
            for m in MODES:
                support = ConditionalMahalanobis.fit(train, max_components=m)
                e = support_values(support, selected)
                u = units(e)
                seed_state["supports"][str(m)] = {"geometry": support.state_dict(), **u}
                selected_sum[m] += ((e - u["center"]) / u["scale"] + 1) / len(seeds)
                geometries[m] = support
            for start in range(0, len(combined), 4096):
                part = (
                    adapter.normalize(features[combined[start : start + 4096]])
                    if mapping is None
                    else mapping.transform(zquery[start : start + 4096])
                )
                part = (part - fmean) / fscale
                for m in MODES:
                    u = seed_state["supports"][str(m)]
                    e = geometries[m].score(part)[0]
                    query_sum[m][start : start + len(part)] += (
                        (e - u["center"]) / u["scale"] + 1
                    ).numpy() / len(seeds)
            state["seeds"].append(seed_state)
            print(
                args.dataset,
                kind,
                "seed",
                seed,
                "scored",
                round(time.perf_counter() - tick, 2),
                flush=True,
            )
        for m in MODES:
            u = units(selected_sum[m])
            state["ensemble_units"][str(m)] = u
            query_sum[m] = (query_sum[m] - u["center"]) / u["scale"] + 1
        torch.save(state, out / f"{kind}_detector.pt")
        restored = H1Detector(torch.load(out / f"{kind}_detector.pt", weights_only=True))
        audit = np.linspace(0, len(combined) - 1, 64, dtype=np.int64)
        actual = restored.score(features[combined[audit]])
        errors = {
            str(m): float(
                np.max(
                    np.abs(actual[m].numpy() - query_sum[m][audit])
                    / (1 + np.abs(query_sum[m][audit]))
                )
            )
            for m in MODES
        }
        if max(errors.values()) > 1e-7:
            raise ValueError(f"Restored scoring differs {kind}: {errors}")
        report["audits"][kind] = {
            "restore_scaled_errors": errors,
            "elapsed_seconds": time.perf_counter() - tick,
        }
        for m in MODES:
            record(f"{kind}_m{m}", query_sum[m])
            if kind == "raw":
                raw_rows[m] = query_sum[m]
            else:
                record(f"raw_plus_{kind}_m{m}", np.maximum(raw_rows[m], query_sum[m]))
        del train, selected, actual, restored
    torch.save(
        {
            "minutes": torch.from_numpy(minutes),
            "labels": torch.from_numpy(labels),
            "types": types,
            "methods": score_archive,
            "source_calibration_minutes": torch.from_numpy(cal_minutes),
        },
        out / "scores.pt",
    )
    torch.save(decisions, out / "decisions.pt")
    report.update(
        status="completed",
        elapsed_seconds=time.perf_counter() - started,
        source_roles_disjoint=True,
        full_target_rows_and_windows=True,
        pca_explained_variance_ratio=adapter.state["explained_variance_ratio"].tolist(),
        source_only_fit=True,
        quantum_execution="CPU exact statevector",
    )
    write_json(out / "report.json", report)
    print("COMPLETE", args.dataset, report["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--dataset", choices=PACKETS, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--reference", type=int, default=4096)
    run(p.parse_args())
