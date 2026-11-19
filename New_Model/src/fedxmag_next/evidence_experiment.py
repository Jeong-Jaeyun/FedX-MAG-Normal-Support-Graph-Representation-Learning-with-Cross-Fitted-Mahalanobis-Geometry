"""Factorial weak initiation and additive evidence, with unchanged prototypes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .causal_risk import scan_contributors
from .experiments import sha256, write_json
from .hierarchical_experiment import minute_groups
from .prototype_guard import SCOPES, entity_keys
from .prototype_temporal import frozen_thresholds, replay_path
from .service_risk_keys import PROTOCOL_COLUMNS, service_keys
from .weak_evidence import CausalEvidenceMemory
from .window_bridge import aggregate_connections
from .window_replay import run as evaluate_windows


def evidence_path(
    values,
    reference,
    ratios,
    minutes,
    groups,
    keys,
    times,
    thresholds,
    scope,
    mode,
    initiation,
    checkpoint,
    *,
    aggregation,
    risk_key_values=None,
    guard_ratios=None,
):
    guard = ratios if guard_ratios is None else np.asarray(guard_ratios)
    if np.shape(guard) != np.shape(ratios):
        raise ValueError("Evidence guard and filtering distances must align")
    q95 = float(np.quantile(reference, 0.95))
    sorted_reference = np.sort(reference)
    memory = CausalEvidenceMemory(q95, mode)
    energies, changed = [], 0
    for i, minute in enumerate(minutes):
        ids = groups.get(int(minute), np.empty(0, dtype=np.int64))
        base = values[ids].copy()
        familiar = guard[ids] <= 1 + 1e-6
        finite = np.isfinite(ratios[ids])
        base[finite] = np.minimum(base[finite], ratios[ids][finite] ** 2 * q95)
        active = (
            entity_keys({key: value[ids] for key, value in keys.items()}, scope)
            if risk_key_values is None
            else [risk_key_values[j] for j in ids]
        )
        shadow = None
        if i == len(minutes) // 2:
            torch.save(memory.state_dict(), checkpoint)
            shadow = CausalEvidenceMemory.from_state_dict(torch.load(checkpoint, weights_only=True))
        predicted = memory.predict((int(minute) + 1) * 60, active, base, times[ids])
        if shadow is not None:
            if not np.array_equal(
                shadow.predict((int(minute) + 1) * 60, active, base, times[ids]), predicted
            ):
                raise ValueError("Actual weak-evidence disk resume changed the next prediction")
            memory = shadow
        predicted[familiar] = base[familiar]
        if initiation == "weak":
            contributors = np.flatnonzero((~familiar) & (base > q95))
        else:
            alarm = (
                len(base)
                and aggregate_connections(
                    base, sorted_reference, aggregation, reference_sorted=True
                )
                > thresholds[i]
            )
            contributors = (
                scan_contributors(base, sorted_reference, aggregation, reference_sorted=True)
                if alarm
                else np.empty(0, dtype=np.int64)
            )
            contributors = contributors[~familiar[contributors]]
        memory.commit(contributors)
        energies.append(
            aggregate_connections(predicted, sorted_reference, aggregation, reference_sorted=True)
            if len(base)
            else 0.0
        )
        changed += int(not np.array_equal(base, predicted))
    return np.array(energies), {
        "changed_windows_by_carry": changed,
        "maximum_keys": memory.maximum_seen,
        "maximum_entries": memory.maximum_entries,
        "midpoint_disk_restore_exact": True,
    }


def run(args):
    torch.set_num_threads(args.workers)
    service_mode = getattr(args, "routing", "endpoint") == "service"
    if service_mode and not args.evidence_control:
        raise ValueError("Service routing requires its matched endpoint-evidence control")
    prototype, control, output = map(Path, (args.prototype, args.control, args.output))
    registration = json.loads((prototype / "registration.json").read_text())
    risk_root = Path(registration["args"]["run"])
    risk_registration = json.loads((risk_root / "registration.json").read_text())
    data_path, key_path = (Path(risk_registration["args"][k]) for k in ("data", "keys"))
    if (
        sha256(data_path) != registration["data_sha256"]
        or sha256(key_path) != registration["keys_sha256"]
    ):
        raise ValueError("Weak-evidence observations or endpoint metadata changed")
    if sha256(risk_root / "row_scores.pt") != registration["row_cache_sha256"]:
        raise ValueError("Weak-evidence original row cache changed")
    cache = torch.load(risk_root / "row_scores.pt", weights_only=True)
    distances = torch.load(prototype / "prototype_distances.pt", weights_only=True)
    if not torch.equal(cache["combined_indices"], distances["combined_indices"]):
        raise ValueError("Weak-evidence ratios do not align with original row energies")
    combined = cache["combined_indices"].numpy()
    with np.load(data_path) as payload:
        times, signature = payload["observed_at"][combined], payload["signature"]
        if service_mode:
            names = payload["feature_names"].tolist()
            columns = [names.index(k) for k in (*PROTOCOL_COLUMNS, "available_dst_port")]
            metadata = payload["features"][:, columns][combined]
    with np.load(key_path) as payload:
        if not np.array_equal(signature, payload["signature"]):
            raise ValueError("Weak-evidence endpoint keys differ")
        keys = {k: payload[k][combined] for k in ("source", "destination")}
        if service_mode:
            ports = payload["port"][combined]
    routed = (
        {
            scope: service_keys({**keys, "port": ports}, metadata[:, :4], metadata[:, 4], scope)
            for scope in SCOPES
        }
        if service_mode
        else {}
    )
    groups = minute_groups(times // 60)
    values, reference = (
        cache["row_scores"][args.head].numpy(),
        cache["row_reference"][args.head].numpy(),
    )
    output.mkdir(parents=True, exist_ok=False)
    common = {
        "args": vars(args),
        "dataset": registration["dataset"],
        "data_sha256": registration["data_sha256"],
        "row_cache_sha256": registration["row_cache_sha256"],
        "normal_view": args.normal_view,
        "aggregation": registration["aggregation"],
        "risk_routing": "observed service" if service_mode else "endpoint",
        "source_geometry_and_normal_distances_fixed": True,
        "risk": "one strongest original contribution per key/window, TTL300/decay300/5000keys",
        "factorial": "peak/sum x window-gated/row-q95-seeded; novel rows only; no label clearing",
    }
    write_json(output / "registration.json", common)
    for capacity in (39, 50):
        source = control if capacity == 39 else control / "capacity50"
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir()
            write_json(destination / "registration.json", {**common, "capacity": capacity})
        checkpoints = destination / "checkpoints"
        checkpoints.mkdir()
        old = torch.load(source / "scores.pt", weights_only=True)
        evidence_control = None
        if service_mode:
            parent_root = Path(args.evidence_control)
            if capacity == 50:
                parent_root = parent_root / "capacity50"
            evidence_control = torch.load(parent_root / "scores.pt", weights_only=True)
            if any(not torch.equal(old[k], evidence_control[k]) for k in ("minutes", "labels")):
                raise ValueError("Service-evidence control changed the target census")
        minutes, labels = old["minutes"].numpy(), old["labels"].numpy()
        baseline = old["methods"][args.head]
        thresholds = frozen_thresholds(
            baseline["energy"].numpy(),
            baseline["observed"].numpy(),
            labels,
            baseline["source_calibration"].tolist(),
            capacity,
        )
        archive = {name: old["methods"][name] for name in ("raw_knn", args.head)}
        diagnostics, matched = {}, []
        for scope in SCOPES:
            ratio = distances["ratios"][scope][args.normal_view].numpy()
            tag = "rawview_" if args.normal_view == "raw" and args.head.startswith("deep_") else ""
            parent_name = f"{args.head}_{tag}continuous_prototype_risk_fixed_{scope}"
            parent_energy, _ = replay_path(
                values,
                reference,
                ratio,
                minutes,
                groups,
                keys,
                times,
                registration["aggregation"],
                thresholds,
                "prototype_risk_fixed",
                scope,
                checkpoints / f"matched_{scope}.pt",
                continuous_filter=True,
            )
            if not np.array_equal(parent_energy, old["methods"][parent_name]["energy"].numpy()):
                raise ValueError(
                    "Weak-evidence matched parent trajectory did not reproduce exactly"
                )
            archive[parent_name] = old["methods"][parent_name]
            matched.append(parent_name)
            for mode in ("peak", "sum"):
                for initiation in ("gated", "weak"):
                    endpoint_head = f"{args.head}_evidence_{mode}_{initiation}_{scope}"
                    head = (
                        endpoint_head.replace("_evidence_", "_service_evidence_", 1)
                        if service_mode
                        else endpoint_head
                    )
                    if service_mode:
                        repeated, _ = evidence_path(
                            values,
                            reference,
                            ratio,
                            minutes,
                            groups,
                            keys,
                            times,
                            thresholds,
                            scope,
                            mode,
                            initiation,
                            checkpoints / f"endpoint_{endpoint_head}.pt",
                            aggregation=registration["aggregation"],
                        )
                        if not np.array_equal(
                            repeated,
                            evidence_control["methods"][f"{endpoint_head}_frozen"][
                                "energy"
                            ].numpy(),
                        ):
                            raise ValueError(
                                "Service split changed its matched endpoint evidence trajectory"
                            )
                        for feedback in ("frozen", "candidate"):
                            matched_name = f"{endpoint_head}_{feedback}"
                            archive[matched_name] = evidence_control["methods"][matched_name]
                    energy, diagnostics[head] = evidence_path(
                        values,
                        reference,
                        ratio,
                        minutes,
                        groups,
                        keys,
                        times,
                        thresholds,
                        scope,
                        mode,
                        initiation,
                        checkpoints / f"{head}.pt",
                        aggregation=registration["aggregation"],
                        risk_key_values=routed[scope] if service_mode else None,
                    )
                    for feedback in ("frozen", "candidate"):
                        archive[f"{head}_{feedback}"] = {
                            **baseline,
                            "energy": torch.from_numpy(energy.copy()),
                            "feedback_energy": baseline["energy"].clone()
                            if feedback == "frozen"
                            else torch.from_numpy(energy.copy()),
                        }
        torch.save({**old, "methods": archive}, destination / "scores.pt")
        evaluate_windows(
            argparse.Namespace(
                run=str(destination),
                output=str(destination / "evaluation"),
                buffer=capacity,
                fpr=0.05,
                history_policy="continuous",
            )
        )
        actual = torch.load(destination / "evaluation/decisions.pt", weights_only=True)
        expected = torch.load(source / "evaluation/decisions.pt", weights_only=True)
        for name in ("raw_knn", args.head, *matched):
            if any(
                not torch.equal(value, actual[name][key]) for key, value in expected[name].items()
            ):
                raise ValueError(f"Weak-evidence matched control decisions changed: {name}")
        if service_mode:
            endpoint_expected = torch.load(
                parent_root / "evaluation/decisions.pt", weights_only=True
            )
            for name in archive:
                if (
                    "_evidence_" in name
                    and "_service_evidence_" not in name
                    and any(
                        not torch.equal(value, actual[name][key])
                        for key, value in endpoint_expected[name].items()
                    )
                ):
                    raise ValueError("Service evidence changed a matched endpoint decision")
        report = json.loads((destination / "evaluation/report.json").read_text())
        write_json(
            destination / "report.json",
            {
                **report,
                "all_matched_parent_trajectories_and_decisions_exact": True,
                "evidence_diagnostics": diagnostics,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prototype", "control", "head", "normal-view", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--routing", choices=("endpoint", "service"), default="endpoint")
    parser.add_argument("--evidence-control")
    run(parser.parse_args())
