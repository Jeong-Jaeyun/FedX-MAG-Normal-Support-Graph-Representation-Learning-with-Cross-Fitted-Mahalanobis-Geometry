"""H2 representation x corruption x scoring experiment, normal-only fitting."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .experiments import sha256, write_json
from .geometry import ConditionalMahalanobis
from .h1_classical_features import SourcePCA
from .h1_experiment import H1Detector, metrics, replay, support_values, units
from .h2_data import H2Data, corruption_audit, negative_rows
from .h2_models import ClassicalMLPEnergy, ReuploadingQuantumEnergy

KINDS = ("quantum", "quantum_noent", "mlp32", "mlp_budget", "full_mlp64")
HEADS = ("scalar", "support")
POLICIES = ("fixed", "delayed_10pct", "immediate")


def make_model(kind, input_dim, seed):
    if kind in ("quantum", "quantum_noent"):
        return ReuploadingQuantumEnergy(seed=seed, entangled=kind == "quantum")
    hidden, depth = {"mlp32": (32, 2), "mlp_budget": (6, 1), "full_mlp64": (64, 3)}[kind]
    return ClassicalMLPEnergy(
        input_dim=input_dim if kind == "full_mlp64" else 8, hidden=hidden, depth=depth, seed=seed
    )


def model_input(adapter, raw, kind):
    return adapter.normalize(raw) if kind == "full_mlp64" else adapter.transform(raw)


@torch.no_grad()
def embeddings(model, x, batch=1024):
    return torch.cat([model.embed(v) for v in x.split(batch)])


@torch.no_grad()
def validation_bce(model, positive, negative):
    total, count = 0.0, 0
    for values, label in ((positive, 0.0), (negative, 1.0)):
        for part in values.split(512):
            logits = model(part)
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite H2 validation logits")
            total += float(
                nn.functional.binary_cross_entropy_with_logits(
                    logits, torch.full_like(logits, label), reduction="sum"
                )
            )
            count += len(part)
    if not np.isfinite(total) or not count:
        raise ValueError("Nonfinite or empty H2 validation loss")
    return total / count


def train_model(
    model,
    kind,
    adapter,
    raw_train,
    raw_validation,
    groups,
    names,
    policy,
    seed,
    epochs=30,
    patience=5,
):
    """The two negative policies share pools, batches and seeds across models."""
    if epochs < 1:
        raise ValueError("H2 training requires at least one epoch")
    positive = model_input(adapter, raw_train, kind)
    val_positive = model_input(adapter, raw_validation, kind)
    val_raw_negative = negative_rows(raw_validation, groups, names, policy, 7189)
    validation_audit = corruption_audit(raw_validation, val_raw_negative, groups, names)
    val_negative = model_input(adapter, val_raw_negative, kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_state = initial_state
    best = validation_bce(model, val_positive, val_negative)
    history = [{"epoch": 0, "selection_synthetic_bce": best}]
    best_epoch, waiting = 0, 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        raw_negative = negative_rows(raw_train, groups, names, policy, seed * 1000 + epoch)
        if epoch == 1:
            train_audit = corruption_audit(raw_train, raw_negative, groups, names)
        negative = model_input(adapter, raw_negative, kind)
        order = torch.randperm(len(positive), generator=generator)
        losses = []
        model.train()
        for ids in order.split(256):
            values = torch.cat((positive[ids], negative[ids]))
            target = torch.cat((torch.zeros(len(ids)), torch.ones(len(ids)))).double()
            loss = nn.functional.binary_cross_entropy_with_logits(model(values), target)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite H2 training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        value = validation_bce(model, val_positive, val_negative)
        history.append(
            {"epoch": epoch, "train_bce": float(np.mean(losses)), "selection_synthetic_bce": value}
        )
        if value < best:
            best, best_epoch, waiting = value, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waiting += 1
        if epoch == 1 or epoch % 5 == 0:
            print(kind, policy, seed, "epoch", epoch, "selection_bce", round(value, 6), flush=True)
        if waiting >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    drift = (
        sum(
            float((best_state[k] - initial_state[k]).square().sum())
            for k in best_state
            if best_state[k].is_floating_point()
        )
        ** 0.5
    )
    return {
        "history": history,
        "selected_epoch": best_epoch,
        "parameter_l2_change": drift,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_corruption_audit": train_audit,
        "validation_corruption_audit": validation_audit,
    }


class H2Detector:
    """Self-contained frozen learned row scorer; accepts features, never labels."""

    def __init__(self, state):
        if state["format_version"] != 1:
            raise ValueError("Unsupported H2 artifact")
        self.state = state
        self.adapter = SourcePCA.from_state_dict(state["adapter"])
        self.models = []
        for seed in state["seeds"]:
            model = make_model(state["kind"], len(state["feature_names"]), seed["seed"])
            model.load_state_dict(seed["model"])
            model.eval()
            self.models.append((model, seed))

    @torch.no_grad()
    def score(self, raw):
        x = model_input(self.adapter, raw, self.state["kind"])
        totals = {h: torch.zeros(len(raw), dtype=torch.float64) for h in HEADS}
        for model, seed in self.models:
            features = embeddings(model, x)
            scores = {"scalar": model.readout(features).squeeze(-1)}
            normalized = (features - seed["feature_mean"]) / seed["feature_scale"]
            scores["support"] = support_values(
                ConditionalMahalanobis.from_state_dict(seed["geometry"]), normalized
            )
            for h in HEADS:
                u = seed["units"][h]
                totals[h] += ((scores[h] - u["center"]) / u["scale"] + 1) / len(self.models)
        for h in HEADS:
            u = self.state["ensemble_units"][h]
            totals[h] = (totals[h] - u["center"]) / u["scale"] + 1
        return totals

    def score_with_raw(self, raw):
        result = self.score(raw)
        raw_score = H1Detector(self.state["raw_bypass_state"]).score(raw)[8]
        result.update({"raw_plus_" + h: torch.maximum(raw_score, result[h]) for h in HEADS})
        return result


def run(args):
    torch.set_num_threads(args.workers)
    torch.set_num_interop_threads(1)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    data = H2Data(args.root, args.dataset, smoke=args.smoke)
    seeds = (17,) if args.smoke else (17, 29, 41)
    epochs = 1 if args.smoke else 30
    report = {
        "status": "running",
        "dataset": args.dataset,
        "smoke": args.smoke,
        "methods": {},
        "audits": {},
        "training": {},
    }
    registration = {
        "args": vars(args),
        "smoke_only": args.smoke,
        **data.hashes,
        "claim": "reused development data, CPU exact simulation, no QPU advantage claim",
        "fit_rows": len(data.fit),
        "validation_rows": len(data.validation),
        "selection_rows": len(data.selection),
        "scored_rows": len(data.combined),
        "target_windows": len(data.minutes),
        "seeds": seeds,
        "max_epochs": epochs,
        "early_stopping": "patience5, smallest source-normal vs synthetic BCE including epoch0",
        "kinds": KINDS,
        "heads": HEADS,
        "corruptions": ["original", "conditional"],
        "normal_memory": False,
        "temporal_evidence": False,
        "aggregation": "top3",
        "calibration_capacity": 50,
        "alpha": 0.05,
        "primary_support_modes": 8,
        "negative_pool": "whole role pool once per epoch, before PCA",
        "implementation_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
    }
    write_json(out / "registration.json", registration)
    torch.save(
        {
            "adapter": data.adapter.state_dict(),
            "fit_indices": torch.from_numpy(data.fit),
            "validation_indices": torch.from_numpy(data.validation),
            "source_role_indices": {k: torch.from_numpy(v) for k, v in data.roles.items()},
            "combined_indices": torch.from_numpy(data.combined),
        },
        out / "source_adapter.pt",
    )
    archives, decisions = {}, {}

    def record(name, windows):
        cal, energy = windows[: len(data.cal_minutes)], windows[len(data.cal_minutes) :]
        archives[name] = {
            "source_calibration": torch.from_numpy(cal),
            "energy": torch.from_numpy(energy),
        }
        report["methods"][name] = {}
        for policy in POLICIES:
            flags, thresholds = replay(energy, cal, data.minutes, data.labels, policy)
            report["methods"][name][policy] = metrics(flags, energy, data.labels, data.types)
            decisions[name + "/" + policy] = {
                "flags": torch.from_numpy(flags),
                "thresholds": torch.from_numpy(thresholds),
            }
        m = report["methods"][name]["immediate"]
        print(args.dataset, name, "immediate", m["tp"], m["fp"], flush=True)
        write_json(out / f"progress_{len(archives):03d}.json", report)

    raw_state = torch.load(data.parent / "raw_detector.pt", weights_only=True)
    raw_detector = H1Detector(raw_state)
    raw_rows = np.empty(len(data.combined))
    for start in range(0, len(raw_rows), 2048):
        raw_rows[start : start + 2048] = raw_detector.score(
            data.features[data.combined[start : start + 2048]]
        )[8].numpy()
    record("raw_m8_reference", data.windows(raw_rows))
    if not args.smoke:
        baseline = data.baselines["frozen_dnn20k_m8"]
        record(
            "frozen_dnn20k_m8_reference",
            np.r_[baseline["source_calibration"].numpy(), baseline["energy"].numpy()],
        )
    for corruption in ("original", "conditional"):
        for kind in KINDS:
            tick = time.perf_counter()
            state = {
                "format_version": 1,
                "kind": kind,
                "corruption": corruption,
                "feature_names": data.names,
                "adapter": data.adapter.state_dict(),
                "seeds": [],
                "ensemble_units": {},
                "raw_bypass_state": raw_state,
            }
            selection_sum = {
                h: torch.zeros(len(data.selection), dtype=torch.float64) for h in HEADS
            }
            query_sum = {h: np.zeros(len(data.combined)) for h in HEADS}
            xfit = model_input(data.adapter, data.features[data.fit], kind)
            xsel = model_input(data.adapter, data.features[data.selection], kind)
            for seed in seeds:
                model = make_model(kind, len(data.names), seed)
                trained = train_model(
                    model,
                    kind,
                    data.adapter,
                    data.features[data.fit],
                    data.features[data.validation],
                    data.groups,
                    data.names,
                    corruption,
                    seed,
                    epochs=epochs,
                )
                report["training"][f"{corruption}/{kind}/{seed}"] = trained
                with torch.no_grad():
                    ffit, fsel = embeddings(model, xfit), embeddings(model, xsel)
                    mean = ffit.mean(0)
                    scale = ffit.std(0, unbiased=False).clamp_min(1e-6)
                    support = ConditionalMahalanobis.fit((ffit - mean) / scale, max_components=8)
                    selected = {
                        "scalar": model.readout(fsel).squeeze(-1),
                        "support": support_values(support, (fsel - mean) / scale),
                    }
                    seed_units = {h: units(selected[h]) for h in HEADS}
                    for h in HEADS:
                        u = seed_units[h]
                        selection_sum[h] += ((selected[h] - u["center"]) / u["scale"] + 1) / len(
                            seeds
                        )
                    for start in range(0, len(data.combined), 2048):
                        ids = data.combined[start : start + 2048]
                        f = model.embed(model_input(data.adapter, data.features[ids], kind))
                        scores = {
                            "scalar": model.readout(f).squeeze(-1),
                            "support": support.score((f - mean) / scale)[0],
                        }
                        for h in HEADS:
                            u = seed_units[h]
                            query_sum[h][start : start + len(ids)] += (
                                (scores[h] - u["center"]) / u["scale"] + 1
                            ).numpy() / len(seeds)
                state["seeds"].append(
                    {
                        "seed": seed,
                        "model": model.state_dict(),
                        "feature_mean": mean,
                        "feature_scale": scale,
                        "geometry": support.state_dict(),
                        "units": seed_units,
                        "training": trained,
                    }
                )
                print(
                    args.dataset,
                    corruption,
                    kind,
                    seed,
                    "trained and scored",
                    round(time.perf_counter() - tick, 1),
                    flush=True,
                )
            for h in HEADS:
                u = units(selection_sum[h])
                state["ensemble_units"][h] = u
                query_sum[h] = (query_sum[h] - u["center"]) / u["scale"] + 1
            path = out / f"{corruption}_{kind}_detector.pt"
            torch.save(state, path)
            restored = H2Detector(torch.load(path, weights_only=True))
            audit = np.linspace(0, len(data.combined) - 1, 64, dtype=np.int64)
            actual = restored.score_with_raw(data.features[data.combined[audit]])
            errors = {
                h: float(
                    np.max(
                        np.abs(actual[h].numpy() - query_sum[h][audit])
                        / (1 + np.abs(query_sum[h][audit]))
                    )
                )
                for h in HEADS
            }
            for h in HEADS:
                expected = np.maximum(raw_rows[audit], query_sum[h][audit])
                errors["raw_plus_" + h] = float(
                    np.max(
                        np.abs(actual["raw_plus_" + h].numpy() - expected) / (1 + np.abs(expected))
                    )
                )
            if max(errors.values()) > 1e-7:
                raise ValueError(f"Restored H2 detector differs: {errors}")
            report["audits"][f"{corruption}/{kind}"] = {
                "restore_scaled_errors": errors,
                "elapsed_seconds": time.perf_counter() - tick,
            }
            for h in HEADS:
                record(f"{corruption}_{kind}_{h}", data.windows(query_sum[h]))
                record(
                    f"raw_plus_{corruption}_{kind}_{h}",
                    data.windows(np.maximum(raw_rows, query_sum[h])),
                )
    torch.save(
        {
            "minutes": torch.from_numpy(data.minutes),
            "labels": torch.from_numpy(data.labels),
            "types": data.types,
            "methods": archives,
            "source_calibration_minutes": torch.from_numpy(data.cal_minutes),
        },
        out / "scores.pt",
    )
    torch.save(decisions, out / "decisions.pt")
    report.update(
        status="completed",
        elapsed_seconds=time.perf_counter() - started,
        source_only_fit=True,
        full_target_rows_and_windows=not args.smoke,
        quantum_execution="CPU exact differentiable statevector",
    )
    write_json(out / "report.json", report)
    print(
        "COMPLETE",
        args.dataset,
        "smoke" if args.smoke else "full",
        report["elapsed_seconds"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", choices=("cic", "xiiotid", "ton"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
