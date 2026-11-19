"""Matched service refinement and finite normal-history capacity for sparse CIC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evidence_experiment import evidence_path
from .experiments import sha256, write_json
from .hierarchical_experiment import minute_groups
from .prototype_guard import SCOPES
from .prototype_temporal import frozen_thresholds
from .service_risk_keys import PROTOCOL_COLUMNS, service_keys
from .sparse_cic_experiment import HEADS
from .window_replay import run as evaluate_windows


def run(args):
    torch.set_num_threads(args.workers)
    control, output = Path(args.control), Path(args.output)
    old_registration = json.loads((control / "registration.json").read_text())
    prototype = Path(old_registration["args"]["prototype"])
    registration = json.loads((prototype / "registration.json").read_text())
    risk_root = Path(registration["args"]["run"])
    risk_registration = json.loads((risk_root / "registration.json").read_text())
    data_path, key_path = (Path(risk_registration["args"][k]) for k in ("data", "keys"))
    if (
        sha256(data_path) != registration["data_sha256"]
        or sha256(key_path) != registration["keys_sha256"]
    ):
        raise ValueError("Service sparse observations or endpoint metadata changed")
    if sha256(risk_root / "row_scores.pt") != registration["row_cache_sha256"]:
        raise ValueError("Service sparse original row energies changed")
    cache = torch.load(risk_root / "row_scores.pt", weights_only=True)
    distances = torch.load(prototype / "prototype_distances.pt", weights_only=True)
    if not torch.equal(cache["combined_indices"], distances["combined_indices"]):
        raise ValueError("Service sparse prototype distances have a different row census")
    combined = cache["combined_indices"].numpy()
    with np.load(data_path) as payload:
        names = payload["feature_names"].tolist()
        fields = payload["features"][
            :, [names.index(k) for k in (*PROTOCOL_COLUMNS, "available_dst_port")]
        ][combined]
        times, signature = payload["observed_at"][combined], payload["signature"]
    with np.load(key_path) as payload:
        if not np.array_equal(signature, payload["signature"]):
            raise ValueError("Service sparse endpoints have different signatures")
        keys = {k: payload[k][combined] for k in ("source", "destination")}
        ports = payload["port"][combined]
    routed = {
        scope: service_keys({**keys, "port": ports}, fields[:, :4], fields[:, 4], scope)
        for scope in SCOPES
    }
    groups = minute_groups(times // 60)
    source39 = torch.load(control / "scores.pt", weights_only=True)
    minutes, labels = source39["minutes"].numpy(), source39["labels"].numpy()
    output.mkdir(parents=True, exist_ok=False)
    common = {
        "args": vars(args),
        "dataset": "cic",
        "data_sha256": registration["data_sha256"],
        "row_cache_sha256": registration["row_cache_sha256"],
        "aggregation": "top3",
        "normal_view": "raw",
        "capacities": [39, 50, 59],
        "change": (
            "service routing and matched59 normal-score history; "
            "fixed alpha.05 and original observations"
        ),
    }
    write_json(output / "registration.json", common)
    for capacity in (39, 50, 59):
        source = control if capacity != 50 else control / "capacity50"
        destination = output if capacity == 39 else output / f"capacity{capacity}"
        if capacity != 39:
            destination.mkdir()
            write_json(destination / "registration.json", {**common, "capacity": capacity})
        checkpoints = destination / "checkpoints"
        checkpoints.mkdir()
        old = torch.load(source / "scores.pt", weights_only=True)
        if any(not torch.equal(old[k], source39[k]) for k in ("minutes", "labels")):
            raise ValueError("Service sparse capacity comparison changed target population")
        static_names = ("raw_knn", *HEADS, *(f"{head}_sparse_top3" for head in HEADS))
        archive = (
            dict(old["methods"])
            if capacity != 59
            else {name: old["methods"][name] for name in static_names}
        )
        diagnostics, endpoint_errors = {}, {}
        for head in HEADS:
            base = old["methods"][f"{head}_sparse_top3"]
            if capacity == 59 and len(base["source_calibration"]) < 59:
                raise ValueError("New59-capacity comparison lacks its complete source calibration")
            thresholds = frozen_thresholds(
                base["energy"].numpy(),
                base["observed"].numpy(),
                labels,
                base["source_calibration"].tolist(),
                capacity,
            )
            values, reference = (
                cache["row_scores"][head].numpy(),
                cache["row_reference"][head].numpy(),
            )
            for scope in SCOPES:
                ratio = distances["ratios"][scope]["raw"].numpy()
                for rule in ("peak_gated", "peak_weak", "sum_weak"):
                    mode, initiation = rule.split("_")
                    parent_name = f"{head}_sparse_top3_{rule}_{scope}"
                    endpoint, endpoint_diag = evidence_path(
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
                        checkpoints / f"endpoint_{parent_name}.pt",
                        aggregation="top3",
                    )
                    if capacity != 59 or initiation == "weak":
                        expected = old["methods"][f"{parent_name}_frozen"]["energy"].numpy()
                        if not np.array_equal(endpoint, expected):
                            raise ValueError("Service sparse matched endpoint trajectory changed")
                        endpoint_errors[parent_name] = 0.0
                    service_name = f"{head}_sparse_top3_service_{rule}_{scope}"
                    child, child_diag = evidence_path(
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
                        checkpoints / f"{service_name}.pt",
                        aggregation="top3",
                        risk_key_values=routed[scope],
                    )
                    diagnostics[parent_name], diagnostics[service_name] = endpoint_diag, child_diag
                    for name, energy in ((parent_name, endpoint), (service_name, child)):
                        for feedback in ("frozen", "candidate"):
                            archive[f"{name}_{feedback}"] = {
                                **base,
                                "energy": torch.from_numpy(energy.copy()),
                                "feedback_energy": base["energy"].clone()
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
        if capacity != 59:
            actual = torch.load(destination / "evaluation/decisions.pt", weights_only=True)
            expected = torch.load(source / "evaluation/decisions.pt", weights_only=True)
            for name, item in expected.items():
                if any(not torch.equal(value, actual[name][key]) for key, value in item.items()):
                    raise ValueError("Service sparse changed a preserved old decision")
        report = json.loads((destination / "evaluation/report.json").read_text())
        write_json(
            destination / "report.json",
            {
                **report,
                "matched_endpoint_energy_errors": endpoint_errors,
                "all_available_parent_decisions_exact": True,
                "regenerated_all_temporal_states_at_own_capacity": True,
                "path_diagnostics": diagnostics,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())
