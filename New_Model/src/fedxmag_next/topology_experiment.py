"""Measure causal normal contact topology and a fixed parent anomaly channel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .complementary_views import metrics, replay
from .experiments import sha256, write_json
from .hierarchical_experiment import minute_groups
from .normal_topology import VIEWS, NormalTopology
from .service_risk_keys import PROTOCOL_COLUMNS
from .window_bridge import aggregate_connections


def run(args):
    prototype, control, output = map(Path, (args.prototype, args.control, args.output))
    protocol = json.loads((prototype / "registration.json").read_text())
    risk_root = Path(protocol["args"]["run"])
    registration = json.loads((risk_root / "registration.json").read_text())
    data_path, key_path = (Path(registration["args"][key]) for key in ("data", "keys"))
    if sha256(data_path) != protocol["data_sha256"] or sha256(key_path) != protocol["keys_sha256"]:
        raise ValueError("Topology input observations or endpoints changed")
    initial = torch.load(prototype / "prototype_risk_detector.pt", weights_only=True)
    with np.load(data_path) as payload:
        names = payload["feature_names"].tolist()
        columns = [names.index(k) for k in (*PROTOCOL_COLUMNS, "available_dst_port")]
        fields = payload["features"][:, columns]
        times, signatures = payload["observed_at"], payload["signature"]
    if not np.isin(fields, (0, 1)).all() or not np.all(fields[:, :4].sum(axis=1) == 1):
        raise ValueError("Topology needs actual protocol and availability observations")
    with np.load(key_path) as payload:
        if not np.array_equal(signatures, payload["signature"]):
            raise ValueError("Topology endpoint signatures differ")
        metadata = [payload[k] for k in ("source", "destination", "port")]
    metadata += [np.argmax(fields[:, :4], axis=1), fields[:, 4].astype(np.int64)]

    def rows_for(ids):
        return list(zip(*(column[ids].tolist() for column in metadata), strict=True))

    reference_ids = initial["source_reference_indices"].numpy()
    reference_ids = reference_ids[np.argsort(times[reference_ids], kind="stable")]
    selection = initial["base_encoder"]["source_role_indices"]["selection"].numpy()
    selection = selection[np.argsort(signatures[selection])[:20000]]
    model = NormalTopology(rows_for(reference_ids), int(times[reference_ids].max()))
    normal_rows = model.score(rows_for(selection))
    row_reference = [np.sort(normal_rows[:, j]) for j in range(len(VIEWS))]
    old = torch.load(control / "scores.pt", weights_only=True)
    parent = old["methods"][args.head]
    cal_minutes = parent["source_calibration_minutes"].numpy()
    cal_ids = np.flatnonzero(np.isin(times // 60, cal_minutes))
    source_roles = initial["base_encoder"]["source_role_indices"]
    if (
        np.intersect1d(reference_ids, cal_ids).size
        or not np.isin(cal_ids, source_roles["calibration"].numpy()).all()
    ):
        raise ValueError("Topology calibration is not the disjoint saved source role")
    if max(times[reference_ids].max(), times[cal_ids].max()) >= int(old["minutes"].min()) * 60:
        raise ValueError("Topology source fitting or calibration entered the target interval")
    normal_cal = model.score(rows_for(cal_ids))
    cal_groups = minute_groups(times[cal_ids] // 60)

    def aggregate(matrix):
        return np.array(
            [
                aggregate_connections(
                    matrix[:, j], row_reference[j], "rank_scan", reference_sorted=True
                )
                if len(matrix)
                else 0.0
                for j in range(len(VIEWS))
            ]
        )

    source_calibration = np.stack(
        [aggregate(normal_cal[cal_groups[int(minute)]]) for minute in cal_minutes]
    )
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": protocol["dataset"],
            "data_sha256": protocol["data_sha256"],
            "keys_sha256": protocol["keys_sha256"],
            "prototype_model_sha256": sha256(prototype / "prototype_risk_detector.pt"),
            "source_reference_indices_exact": True,
            "source_role_order": (
                "original disjoint offline source roles retained; calibration may precede train"
            ),
            "fixed_budget": {"normal_events": 5000, "shrinkage_strength": 20},
            "label_use": "whole verified-normal feedback after current prediction only",
        },
    )
    torch.save(
        {
            "format_version": 1,
            "initial_topology": model.state_dict(),
            "row_reference": [torch.from_numpy(value) for value in row_reference],
            "source_calibration": torch.from_numpy(source_calibration),
            "source_calibration_minutes": torch.from_numpy(cal_minutes),
        },
        output / "topology_detector.pt",
    )
    minutes, labels = old["minutes"].numpy(), old["labels"].numpy()
    groups = minute_groups(times // 60)
    target, restored = [], False
    for i, minute in enumerate(minutes):
        ids = groups.get(int(minute), np.empty(0, dtype=np.int64))
        rows = rows_for(ids)
        now = (int(minute) + 1) * 60
        shadow = None
        if i == len(minutes) // 2:
            torch.save(model.state_dict(), output / "midpoint.pt")
            shadow = NormalTopology.from_state_dict(
                torch.load(output / "midpoint.pt", weights_only=True)
            )
        values = model.query(now, rows, times[ids])
        if shadow is not None:
            if not np.array_equal(shadow.query(now, rows, times[ids]), values):
                raise ValueError("Actual topology midpoint disk restore changed next scores")
            shadow.feedback(verified_normal=bool(labels[i] == 0))
            restored = True
        model.feedback(verified_normal=bool(labels[i] == 0))
        if shadow is not None:
            model = shadow
        target.append(aggregate(values))
        if i % 500 == 0:
            print("topology windows", protocol["dataset"], i, len(minutes), flush=True)
    target = np.stack(target)
    for capacity in (39, 50):
        source = control if capacity == 39 else control / "capacity50"
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir()
        archive = torch.load(source / "scores.pt", weights_only=True)
        if not all(torch.equal(archive[k], old[k]) for k in ("minutes", "labels")):
            raise ValueError("Topology parent windows differ between capacities")
        parent = archive["methods"][args.head]
        if not np.array_equal(parent["source_calibration_minutes"].numpy(), cal_minutes):
            raise ValueError("Topology parent source-calibration population changed")
        original = parent["energy"].numpy()
        feedback = parent.get("feedback_energy", parent["energy"]).numpy()
        normal_cal = parent["source_calibration"].numpy()
        observed = parent["observed"].numpy()
        flags, valid, _ = replay(
            normal_cal[:, None], original[:, None], feedback[:, None], observed, labels, capacity
        )
        expected = torch.load(source / "evaluation/decisions.pt", weights_only=True)[args.head]
        if not np.array_equal(flags, expected["flags"].numpy()) or not np.array_equal(
            valid, expected["valid"].numpy()
        ):
            raise ValueError("Topology single-parent replay did not reproduce its exact decisions")
        reports = {"preserved_parent": metrics(flags, valid, labels, archive["types"])}
        predictions = {}
        for j, view in enumerate(VIEWS):
            for fuse in (False, True):
                calibration = source_calibration[:, j, None]
                query = target[:, j, None]
                history = query
                if fuse:
                    calibration = np.column_stack((normal_cal, calibration))
                    query = np.column_stack((original, query))
                    history = np.column_stack((feedback, history))
                name = view + ("_with_parent" if fuse else "_alone")
                flags, valid, pvalues = replay(
                    calibration, query, history, observed, labels, capacity
                )
                reports[name] = metrics(flags, valid, labels, archive["types"])
                predictions[name] = {
                    "flags": torch.from_numpy(flags),
                    "valid": torch.from_numpy(valid),
                    "pvalue": torch.from_numpy(pvalues),
                }
                print(capacity, name, {k: reports[name][k] for k in ("tpr", "fpr")}, flush=True)
        torch.save(
            {
                "minutes": old["minutes"],
                "labels": old["labels"],
                "types": old["types"],
                "topology_energy": torch.from_numpy(target),
                "methods": predictions,
            },
            destination / "scores.pt",
        )
        write_json(
            destination / "report.json",
            {
                "dataset": protocol["dataset"],
                "status": "causal_normal_topology_development",
                "capacity": capacity,
                "all_parent_decisions_exact": True,
                "actual_midpoint_disk_restore_exact": restored,
                "methods": reports,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prototype", "control", "head", "output"):
        parser.add_argument(f"--{name}", required=True)
    run(parser.parse_args())
