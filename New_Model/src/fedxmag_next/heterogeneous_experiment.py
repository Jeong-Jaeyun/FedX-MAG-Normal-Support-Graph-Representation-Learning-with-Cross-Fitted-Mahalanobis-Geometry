"""DNN, IsolationForest and raw Gaussian causal window consensus controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .complementary_views import metrics
from .experiments import sha256, write_json
from .rank_consensus import RankConsensus

ROOT = Path("New_Model/runs")
FOREST_HEAD = "raw_isolation_forest_samples2048_continuous_prototype_risk_fixed_{scope}_candidate"
RAW_HEAD = "raw_mahalanobis_m1_continuous_prototype_source"
COMPONENTS = {
    "cic": {
        "neural": (
            "round219_cic_current_normal_view",
            "deep_mahalanobis_m1_ensemble_current_guard_service_sum_weak_source_candidate",
        ),
        "forest": ("round208_cic_isolation_forest", FOREST_HEAD.format(scope="source")),
        "gaussian": ("round192_cic_packet_http_cross_view", RAW_HEAD),
    },
    "ton": {
        "neural": (
            "round179_ton_cross_view",
            "deep_mahalanobis_m1_ensemble_rawview_continuous_prototype_risk_fixed_source",
        ),
        "forest": ("round207_ton_isolation_forest", FOREST_HEAD.format(scope="source")),
        "gaussian": ("round173_ton_continuous_prototype", RAW_HEAD),
    },
    "xiiotid": {
        "neural": (
            "round212_xiiotid_service_evidence",
            "deep_mahalanobis_m8_ensemble_service_evidence_sum_weak_peer_frozen",
        ),
        "forest": ("round209_xiiotid_isolation_forest", FOREST_HEAD.format(scope="peer")),
        "gaussian": ("round175_xiiotid_continuous_prototype", RAW_HEAD),
    },
}
DESIGNS = {
    "neural_forest": ("neural", "forest"),
    "neural_gaussian": ("neural", "gaussian"),
    "neural_forest_gaussian": ("neural", "forest", "gaussian"),
}


def member_provenance(folder):
    source = ROOT / folder
    registration = json.loads((source / "registration.json").read_text())
    evidence = {"registration_sha256": sha256(source / "registration.json")}
    if "data_sha256" not in registration:
        parent = Path(registration["args"]["run"])
        if sha256(parent / "prototype_risk_detector.pt") != registration["parent_model_sha256"]:
            raise ValueError("Heterogeneous member's data-provenance parent was changed")
        ancestor = json.loads((parent / "registration.json").read_text())
        if ancestor["dataset"] != registration["dataset"]:
            raise ValueError("Heterogeneous data-provenance parent has a different dataset")
        evidence["ancestor_registration_sha256"] = sha256(parent / "registration.json")
        evidence["parent_model_sha256"] = registration["parent_model_sha256"]
        registration = {**registration, "data_sha256": ancestor["data_sha256"]}
    return registration, evidence


def replay(calibration, query, feedback, observed, labels, minutes, capacity, pooling, checkpoint):
    model = RankConsensus(calibration, capacity, pooling)
    flags, valid = np.zeros(len(query), dtype=bool), np.zeros(len(query), dtype=bool)
    pvalues = np.ones(len(query))
    for i, minute in enumerate(minutes):
        shadow = None
        if checkpoint is not None and i == len(query) // 2:
            torch.save(model.state_dict(), checkpoint)
            shadow = RankConsensus.from_state_dict(torch.load(checkpoint, weights_only=True))
        result = model.query(int(minute), query[i], bool(observed[i]))
        if shadow is not None:
            if shadow.query(int(minute), query[i], bool(observed[i])) != result:
                raise ValueError("Actual heterogeneous midpoint restore changed a prediction")
            model = shadow
        flags[i], valid[i], pvalues[i] = result
        model.feedback(verified_normal=bool(labels[i] == 0), scores=feedback[i])
    return flags, valid, pvalues


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for name, (folder, head) in COMPONENTS[args.dataset].items():
        registration, evidence = member_provenance(folder)
        if registration["dataset"] != args.dataset:
            raise ValueError("Heterogeneous member has a different dataset")
        inputs[name] = {
            "folder": folder,
            "head": head,
            "data_sha256": registration["data_sha256"],
            **evidence,
        }
    if len({item["data_sha256"] for item in inputs.values()}) != 1:
        raise ValueError("Heterogeneous members were not fitted on the same observations")
    for capacity in (39, 50):
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir()
        checkpoints = destination / "checkpoints"
        checkpoints.mkdir()
        members, decisions, provenance, common = {}, {}, {}, None
        origin = None
        for name, (folder, head) in COMPONENTS[args.dataset].items():
            source = ROOT / folder
            if capacity == 50:
                source = source / "capacity50"
            stored = torch.load(source / "scores.pt", weights_only=True)
            if origin is None:
                origin = stored
            elif any(not torch.equal(origin[k], stored[k]) for k in ("minutes", "labels")):
                raise ValueError("Heterogeneous members have a different complete target census")
            if origin["types"] != stored["types"]:
                raise ValueError("Heterogeneous members have different attack subtype annotations")
            members[name] = stored["methods"][head]
            decisions[name] = torch.load(source / "evaluation/decisions.pt", weights_only=True)[
                head
            ]
            times = members[name]["source_calibration_minutes"].numpy()
            common = times if common is None else np.intersect1d(common, times)
            provenance[name] = {
                **inputs[name],
                "scores_sha256": sha256(source / "scores.pt"),
                "decisions_sha256": sha256(source / "evaluation/decisions.pt"),
            }
        if len(common) < 19 or common.max() >= int(origin["minutes"].min()):
            raise ValueError("Heterogeneous fusion needs earlier independent normal calibration")
        cal, query, feedback = {}, {}, {}
        labels, minutes = origin["labels"].numpy(), origin["minutes"].numpy()
        controls, predictions, reports = {}, {}, {}
        for name, item in members.items():
            times = item["source_calibration_minutes"].numpy()
            if not np.array_equal(times[-capacity:], common[-capacity:]):
                raise ValueError("Common calibration would change a single-member control history")
            cal[name] = item["source_calibration"].numpy()[np.searchsorted(times, common)]
            query[name] = item["energy"].numpy()
            feedback[name] = item.get("feedback_energy", item["energy"]).numpy()
            result = replay(
                cal[name][:, None],
                query[name][:, None],
                feedback[name][:, None],
                item["observed"].numpy(),
                labels,
                minutes,
                capacity,
                "any",
                None,
            )
            flags, valid, pvalues = result
            if any(
                not np.array_equal(actual, decisions[name][field].numpy())
                for actual, field in ((flags, "flags"), (valid, "valid"))
            ):
                raise ValueError(f"Heterogeneous single-member control changed: {name}")
            controls[name] = metrics(flags, valid, labels, origin["types"])
            predictions[f"control_{name}"] = {
                "flags": torch.from_numpy(flags),
                "valid": torch.from_numpy(valid),
                "pvalue": torch.from_numpy(pvalues),
            }
        diversity = {}
        for left, right in (("neural", "forest"), ("neural", "gaussian"), ("forest", "gaussian")):
            lf, rf = (predictions[f"control_{name}"]["flags"].numpy() for name in (left, right))
            normal, attack = labels == 0, labels == 1
            diversity[f"{left}_{right}"] = {
                "shared_false_alarms": int((lf & rf & normal).sum()),
                "left_only_false_alarms": int((lf & ~rf & normal).sum()),
                "right_only_false_alarms": int((~lf & rf & normal).sum()),
                "shared_detected_attacks": int((lf & rf & attack).sum()),
                "left_only_detected_attacks": int((lf & ~rf & attack).sum()),
                "right_only_detected_attacks": int((~lf & rf & attack).sum()),
            }
        for design, names in DESIGNS.items():
            calibration = np.column_stack([cal[name] for name in names])
            queries = np.column_stack([query[name] for name in names])
            native = np.column_stack([feedback[name] for name in names])
            observed = np.logical_and.reduce([members[name]["observed"].numpy() for name in names])
            rules = ("any", "mean", "all") if len(names) == 2 else ("any", "mean", "median", "all")
            for rule in rules:
                for policy in ("component_native", "joint_candidate"):
                    name = f"heterogeneous_{design}_{rule}_{policy}"
                    flags, valid, pvalues = replay(
                        calibration,
                        queries,
                        native if policy == "component_native" else queries,
                        observed,
                        labels,
                        minutes,
                        capacity,
                        rule,
                        checkpoints / f"{name}.pt",
                    )
                    reports[name] = metrics(flags, valid, labels, origin["types"])
                    predictions[name] = {
                        "flags": torch.from_numpy(flags),
                        "valid": torch.from_numpy(valid),
                        "pvalue": torch.from_numpy(pvalues),
                    }
                    print(capacity, name, {k: reports[name][k] for k in ("tpr", "fpr")}, flush=True)
        write_json(
            destination / "registration.json",
            {
                "args": vars(args),
                "dataset": args.dataset,
                "capacity": capacity,
                "members": provenance,
                "designs": DESIGNS,
                "common_source_calibration_minutes": common.tolist(),
                "fit": "none; frozen members, symmetric pooled ranks; no target-fitted weights",
                "feedback": (
                    "query before external whole-normal verification; no alarm-based admission"
                ),
            },
        )
        write_json(
            destination / "report.json",
            {
                "dataset": args.dataset,
                "status": "heterogeneous_consensus_development",
                "all_single_member_flags_and_coverage_exact": True,
                "all_actual_consensus_midpoint_restores_exact": True,
                "independent_confirmation": False,
                "controls": controls,
                "component_error_overlap": diversity,
                "methods": reports,
            },
        )
        torch.save(
            {
                "minutes": origin["minutes"],
                "labels": origin["labels"],
                "types": origin["types"],
                "methods": predictions,
            },
            destination / "decisions.pt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=tuple(COMPONENTS))
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
