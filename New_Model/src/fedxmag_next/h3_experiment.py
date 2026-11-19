"""H3 causal eight-minute prediction experiment with disjoint source observations."""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .experiments import write_json
from .geometry import ConditionalMahalanobis
from .h1_classical_features import SourcePCA
from .h1_experiment import metrics, replay, support_values, units
from .h3_data import CONTEXT_MINUTES, HISTORY, ROLES, histories, load_inputs
from .h3_models import METHODS, SequentialMap, fit_gru, predict_gru, ridge_fit, ridge_predict

MODES = (1, 8)
POLICIES = ("fixed", "delayed_10pct", "immediate")


def make_examples(inputs, adapter, query_minutes):
    """Transform a timeline using a frozen normal-source adapter."""
    z = adapter.transform(inputs["summary"]).numpy()
    z[~inputs["available"]] = 0
    x, mask = histories(query_minutes, inputs["minutes"], z, inputs["available"])
    indices = np.searchsorted(inputs["minutes"], query_minutes)
    return dict(x=x, mask=mask, y=z[indices], present=inputs["available"][indices].astype(float))


def represented(state, examples):
    kind = state["kind"]
    x, mask, y = examples["x"], examples["mask"], examples["y"]
    if kind == "delay_support":
        values = np.c_[y, x.reshape(len(x), -1), mask]
    elif kind == "gru":
        values = y - predict_gru(state["predictor"], x, mask)
    else:
        mapping = SequentialMap(kind, state["seed"], state["map"])
        values = y - ridge_predict(state["predictor"], mapping.transform(x, mask))
    return torch.from_numpy(np.c_[values, examples["present"]]).double()


def seed_scores(state, examples):
    values = (represented(state, examples) - state["feature_mean"]) / state["feature_scale"]
    result = {}
    for mode in MODES:
        spec = state["supports"][str(mode)]
        energy = support_values(ConditionalMahalanobis.from_state_dict(spec["geometry"]), values)
        result[mode] = (energy - spec["center"]) / spec["scale"] + 1
    return result


class H3Detector:
    """Restore a frozen predictor and score current summaries with explicit past slots."""

    def __init__(self, state):
        if state["format_version"] != 1:
            raise ValueError("Unsupported H3 artifact")
        self.state = state
        self.adapter = SourcePCA.from_state_dict(state["adapter"])

    def score_examples(self, examples):
        scores = [seed_scores(seed, examples) for seed in self.state["seeds"]]
        return {mode: torch.stack([s[mode] for s in scores]).mean(0) for mode in MODES}

    def score_timeline(self, minutes, summaries, available, query_minutes):
        """Every query requires its current summary row, and missing rows carry available=False."""
        minutes = np.asarray(minutes, dtype=np.int64)
        query_minutes = np.asarray(query_minutes, dtype=np.int64)
        if np.any(np.diff(minutes) <= 0) or not np.isin(query_minutes, minutes).all():
            raise ValueError("Sorted unique timeline must contain every query slot")
        examples = make_examples(
            dict(minutes=minutes, summary=np.asarray(summaries), available=np.asarray(available)),
            self.adapter,
            query_minutes,
        )
        return self.score_examples(examples)


def post_attack_metrics(flags, labels, minutes, past_attack):
    normal = labels == 0
    recent = normal & past_attack
    quiet = normal & ~past_attack
    result = {}
    for name, mask in (("normal_with_attack_in_previous8min", recent), ("other_normal", quiet)):
        result[name] = {
            "windows": int(mask.sum()),
            "false_alarms": int(flags[mask].sum()),
            "fpr": float(flags[mask].mean()) if mask.any() else None,
        }
    runs = []
    for i in range(1, len(minutes)):
        if labels[i] != 0 or labels[i - 1] != 1 or minutes[i] != minutes[i - 1] + 1:
            continue
        end = i + 1
        while end < len(minutes) and labels[end] == 0 and minutes[end] == minutes[end - 1] + 1:
            end += 1
        clears = np.flatnonzero(~flags[i:end])
        runs.append(
            {
                "first_normal_minute": int(minutes[i]),
                "observed_normal_run_minutes": end - i,
                "first_clear_after_minutes": int(clears[0]) if len(clears) else None,
                "right_censored": not bool(len(clears)),
            }
        )
    result["contiguous_attack_to_normal_runs"] = runs
    return result


def run(args):
    torch.set_num_threads(args.workers)
    torch.set_num_interop_threads(1)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    inputs = load_inputs(args.root, args.dataset)
    target_minutes, labels, types = inputs["target_minutes"], inputs["labels"], inputs["types"]
    past_attack = inputs["past_attack"]
    if args.smoke:
        # Exercise all class-dependent metric paths while bounding the query count.
        keep = np.sort(np.r_[np.flatnonzero(labels == 0)[:8], np.flatnonzero(labels == 1)[:8]])
        target_minutes, labels = target_minutes[keep], labels[keep]
        types, past_attack = [types[i] for i in keep], past_attack[keep]
    parents, role_minutes = inputs["parents"], inputs["examples"]
    adapter_minutes = np.unique(role_minutes["train"][:, None] - np.arange(HISTORY + 1))
    train_rows = np.searchsorted(inputs["minutes"], adapter_minutes)
    adapter = SourcePCA.fit(torch.from_numpy(inputs["summary"][train_rows]))
    examples = {role: make_examples(inputs, adapter, role_minutes[role]) for role in ROLES}
    examples["target"] = make_examples(inputs, adapter, target_minutes)
    if args.smoke:
        for role in ("train", "selection"):
            examples[role] = {k: v[:64] for k, v in examples[role].items()}
    seeds = (17,) if args.smoke else (17, 29, 41)
    epochs = 2 if args.smoke else 100
    registration = {
        "status": "registered",
        "args": vars(args),
        "scope": "integration_smoke" if args.smoke else "full_prepared_target_census",
        "input_audit": inputs["audit"],
        "methods": METHODS,
        "seeds": seeds,
        "modes": MODES,
        "primary_modes": 8,
        "feedback_policies": POLICIES,
        "source_role_rule": (
            "maximum feasible55/15/15/15 complete normal-sequence quotas, minimum32 per role; "
            "chronological parent ownership with all14 history/current/context slots inside role"
        ),
        "feature_context_minutes": CONTEXT_MINUTES,
        "normal_window_buffer": 50,
        "alpha": 0.05,
        "gru_epochs": epochs,
        "target_windows_scored": len(target_minutes),
        "target_cold_start_windows": int((examples["target"]["mask"].sum(1) < HISTORY).sum()),
        "target_missing_current": int((examples["target"]["present"] == 0).sum()),
        "quantum_execution": "CPU exact statevector",
    }
    write_json(output / "registration.json", registration)
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "adapter_fit_minutes": torch.from_numpy(adapter_minutes),
            "parent_minutes": {k: torch.from_numpy(v) for k, v in parents.items()},
            "example_minutes": {k: torch.from_numpy(v) for k, v in role_minutes.items()},
            "observation_ids": {k: torch.from_numpy(v) for k, v in inputs["row_ids"].items()},
            "target_minutes": torch.from_numpy(target_minutes),
            "target_history_mask": torch.from_numpy(examples["target"]["mask"]),
        },
        output / "source_adapter.pt",
    )
    print(args.dataset, "H3 source counts", inputs["audit"]["example_counts"], flush=True)
    scores, decisions, report = {}, {}, {"dataset": args.dataset, "methods": {}, "audits": {}}
    for kind in METHODS:
        tick = time.perf_counter()
        artifact = {
            "format_version": 1,
            "kind": kind,
            "adapter": adapter.state_dict(),
            "seeds": [],
            "summary_definition": "prepared-row mean,SD,log1p-count,presence",
            "feature_names": inputs["audit"]["feature_names"],
        }
        per_role = {role: {mode: [] for mode in MODES} for role in ("calibration", "target")}
        for seed in seeds:
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
                    epochs,
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
            reference = represented(state, examples["residual_support"])
            selected = represented(state, examples["selection"])
            state["feature_mean"] = reference.mean(0)
            state["feature_scale"] = reference.std(0, unbiased=False).clamp_min(0.05)
            reference = (reference - state["feature_mean"]) / state["feature_scale"]
            selected = (selected - state["feature_mean"]) / state["feature_scale"]
            state["supports"] = {}
            for mode in MODES:
                support = ConditionalMahalanobis.fit(reference, max_components=mode)
                score_units = units(support_values(support, selected))
                state["supports"][str(mode)] = {"geometry": support.state_dict(), **score_units}
            for role in per_role:
                calculated = seed_scores(state, examples[role])
                for mode in MODES:
                    per_role[role][mode].append(calculated[mode])
            artifact["seeds"].append(state)
            print(
                args.dataset,
                kind,
                "seed",
                seed,
                "elapsed",
                round(time.perf_counter() - tick, 2),
                flush=True,
            )
        torch.save(artifact, output / f"{kind}_detector.pt")
        restored = H3Detector(torch.load(output / f"{kind}_detector.pt", weights_only=True))
        check = np.linspace(0, len(target_minutes) - 1, min(32, len(target_minutes)), dtype=int)
        actual = restored.score_timeline(
            inputs["minutes"], inputs["summary"], inputs["available"], target_minutes[check]
        )
        restore_errors = {}
        for mode in MODES:
            name = f"{kind}_m{mode}"
            energy = torch.stack(per_role["target"][mode]).mean(0).numpy()
            calibration = torch.stack(per_role["calibration"][mode]).mean(0).numpy()
            error = float(
                np.max(np.abs(actual[mode].numpy() - energy[check]) / (1 + np.abs(energy[check])))
            )
            restore_errors[str(mode)] = error
            if error > 1e-6:
                raise ValueError(f"H3 restored scoring differs {kind}: {error}")
            scores[name] = {
                "energy": torch.from_numpy(energy),
                "source_calibration": torch.from_numpy(calibration),
            }
            report["methods"][name] = {}
            for policy in POLICIES:
                flags, thresholds = replay(energy, calibration, target_minutes, labels, policy)
                result = metrics(flags, energy, labels, types)
                result["post_attack"] = post_attack_metrics(
                    flags, labels, target_minutes, past_attack
                )
                report["methods"][name][policy] = result
                decisions[name + "/" + policy] = {
                    "flags": torch.from_numpy(flags),
                    "thresholds": torch.from_numpy(thresholds),
                }
            entry = report["methods"][name]["immediate"]
            print(name, "immediate TP/FP", entry["tp"], entry["fp"], flush=True)
        report["audits"][kind] = {
            "restore_scaled_errors": restore_errors,
            "elapsed_seconds": time.perf_counter() - tick,
            "prediction_fit": [
                {
                    k: seed["predictor"][k]
                    for k in ("validation_mse", "ridge", "selected_epoch")
                    if k in seed["predictor"]
                }
                for seed in artifact["seeds"]
                if "predictor" in seed
            ],
        }
        write_json(output / f"progress_{len(report['audits']):03d}.json", report)
    torch.save(
        {
            "minutes": torch.from_numpy(target_minutes),
            "labels": torch.from_numpy(labels),
            "types": types,
            "source_calibration_minutes": torch.from_numpy(role_minutes["calibration"]),
            "methods": scores,
            "past_attack": torch.from_numpy(past_attack),
        },
        output / "scores.pt",
    )
    torch.save(decisions, output / "decisions.pt")
    report.update(
        status="completed",
        scope=registration["scope"],
        elapsed_seconds=time.perf_counter() - started,
        source_observations_disjoint=True,
        target_windows=len(target_minutes),
        target_cold_start_windows=registration["target_cold_start_windows"],
        target_missing_current=registration["target_missing_current"],
        source_counts=inputs["audit"]["example_counts"],
        pca_explained_variance_ratio=adapter.state["explained_variance_ratio"].tolist(),
    )
    write_json(output / "report.json", report)
    print("H3 COMPLETE", args.dataset, report["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", choices=("cic", "xiiotid", "ton"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
