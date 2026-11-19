"""Pre-registered audit-band confirmation of entity-level normal-support scoring.

``register`` binds the protocol, code, data, and every encoder checkpoint by
SHA-256 without opening any data. ``evaluate`` runs once: it scores the
untouched audit band, applies the pre-declared hypotheses, and runs the
consistency checks (C2: pooled thresholds reproduce the frozen M1 geometry
report; C3: the primary replayed on the consumed test reproduces the
exploratory numbers).
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fedxmag.config import config_fingerprint, feature_groups_from_config, load_stage1_config
from fedxmag.confirmation import model_builders, sha256_file
from fedxmag.m1.detection import (
    MahalanobisDetector,
    auroc,
    cluster_bootstrap_detection,
    conformal_threshold,
    descriptor_matrix,
    detection_summary,
)
from fedxmag.m1.entity import (
    EntityScorer,
    StateGraphModel,
    aggregate,
    encode_control_entities,
    encode_entities,
    localization_rate,
    raw_entity_features,
    stratified_aurocs,
)
from fedxmag.m1.training import embed_clean
from fedxmag.serialization import load_graph_samples

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "configs" / "entity_scoring_confirmation_protocol.yaml"
DEFAULT_OUTPUT = ROOT / "output" / "entity_confirmation"
V3B_CONFIG_SHA256 = "760adb6c6c856baa84885e4c63b21623cec4bce7ce2c354dedfce2413b3efcb5"
CODE_FILES = (
    "src/fedxmag/m1/entity.py",
    "src/fedxmag/m1/detection.py",
    "src/fedxmag/m1/covariance.py",
    "src/fedxmag/m1/training.py",
    "src/fedxmag/confirmation.py",
    "scripts/run_entity_confirmation.py",
    "scripts/build_audit_band.py",
)
EVAL_ROLES = ("calibration", "confirm_normal", "confirm_attack")
REPLAY_ROLES = ("replay_test_normal", "replay_test_attack")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def resolve(protocol: dict[str, Any], reference: str) -> Path:
    prefix, name = reference.split("/", 1)
    key = "prepared_dir" if prefix == "prepared" else "audit_band_dir"
    return ROOT / protocol["data"][key] / name


def checkpoint_paths(protocol: dict[str, Any]) -> dict[str, dict[str, Path]]:
    return {
        name: {str(seed): ROOT / spec["pattern"].format(seed=seed) for seed in spec["seeds"]}
        for name, spec in protocol["encoders"].items()
    }


def bindings(protocol: dict[str, Any], protocol_path: Path) -> dict[str, Any]:
    return {
        "protocol_sha256": sha256_file(protocol_path),
        "code": {name: sha256_file(ROOT / name) for name in CODE_FILES},
        "data": {
            ref: sha256_file(resolve(protocol, ref)) for ref in protocol["data"]["file_sha256"]
        },
        "checkpoints": {
            f"{name}#{seed}": sha256_file(path)
            for name, members in checkpoint_paths(protocol).items()
            for seed, path in members.items()
        },
    }


def register(protocol: dict[str, Any], protocol_path: Path, output: Path) -> None:
    config = load_stage1_config(ROOT / protocol["base_config"])
    if config_fingerprint(config) != V3B_CONFIG_SHA256:
        raise ValueError("Base configuration is not the frozen v3b configuration")
    bound = bindings(protocol, protocol_path)
    mismatched = {ref: h for ref, h in bound["data"].items()
                  if h != protocol["data"]["file_sha256"][ref]}
    if mismatched:
        raise ValueError(f"Data files differ from the protocol: {mismatched}")
    manifest_path = ROOT / protocol["data"]["audit_band_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest["audit_normal_rebuilt_identical"]:
        raise ValueError("C1 failed: audit-band normal windows were not reproduced")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "registration.json"
    if path.exists():
        raise FileExistsError("The confirmation protocol is already registered")
    path.write_text(json.dumps({"status": "registered", "registered_at_utc": now(),
                                "C1_audit_normal_reproduced": True, **bound}, indent=2))
    print(json.dumps({"status": "registered", "protocol_sha256": bound["protocol_sha256"]}))


def load_encoder(kind: str, path: Path, config: dict[str, Any], groups: Any) -> Any:
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    if kind == "graph":
        model_config = config["model"]
        model = StateGraphModel(
            groups.node_dim, groups.edge_dim, hidden_dim=int(model_config["hidden_dim"]),
            graph_dim=int(model_config["graph_dim"]), num_layers=int(model_config["num_layers"]),
            dropout=float(model_config["dropout"]),
        )
    else:
        model = model_builders(config, groups)[kind]()
    model.load_state_dict(state)
    return model.eval()


def evaluate(protocol: dict[str, Any], protocol_path: Path, output: Path) -> None:
    registration = json.loads((output / "registration.json").read_text())
    current = bindings(protocol, protocol_path)
    for key in ("protocol_sha256", "code", "data", "checkpoints"):
        if registration[key] != current[key]:
            raise ValueError(f"{key} changed after registration")
    report_path = output / "confirmation_report.json"
    if report_path.exists():
        raise FileExistsError("The audit band has already been evaluated")
    config = load_stage1_config(ROOT / protocol["base_config"])
    groups = feature_groups_from_config(config)
    roles = protocol["data"]["roles"]
    opened_at = now()
    samples = {
        role: load_graph_samples(resolve(protocol, roles[role]), groups)
        for role in ("fit", *EVAL_ROLES, "replay_test_normal", "replay_test_attack")
    }
    windows = {(w["session"], w["window_start_ns"]): w for w in json.loads(
        resolve(protocol, roles["confirm_attack_windows"]).read_text())}
    attack_info = [windows[(s.metadata["session_key"], s.metadata["window_start_ns"])]
                   for s in samples["confirm_attack"]]
    targets = [float(t) for t in protocol["scoring"]["fpr_targets"]]
    batch = int(protocol["scoring"]["embedding_batch_size"])
    def node_counts(role: str) -> np.ndarray:
        return np.asarray([s.node_features.shape[0] for s in samples[role]], float)

    def dates(role: str) -> np.ndarray:
        return np.asarray([s.metadata["session_key"] for s in samples[role]])

    strata_inputs = {
        "normal_nodes": node_counts("confirm_normal"),
        "attack_nodes": node_counts("confirm_attack"),
        "normal_dates": dates("confirm_normal"),
        "attack_dates": dates("confirm_attack"),
        "attack_fraction": np.asarray([w["attack_fraction"] for w in attack_info]),
    }

    scores: dict[str, dict[str, np.ndarray]] = {}
    members: dict[str, list[str]] = {}
    per_member: dict[str, dict[str, Any]] = {}
    replay: dict[str, float] = {}
    pooled_thresholds: dict[str, dict[str, float]] = {}

    def record(method: str, member: str, graph: dict[str, np.ndarray], extra: dict | None = None):
        summary = detection_summary(graph["calibration"], graph["confirm_normal"],
                                    graph["confirm_attack"], targets)
        summary.update(stratified_aurocs(graph["confirm_normal"], graph["confirm_attack"],
                                         **strata_inputs))
        if extra:
            summary.update(extra)
        per_member.setdefault(method, {})[member] = summary
        key = f"{method}#{member}"
        scores[key] = {"calibration": graph["calibration"], "normal": graph["confirm_normal"],
                       "attack": graph["confirm_attack"]}
        members.setdefault(method, []).append(key)

    paths = checkpoint_paths(protocol)
    entity_specs = [tuple(item) for item in protocol["entity_methods"]]
    needed = {name for name, _, _ in entity_specs} | {
        spec["encoder"] for spec in protocol["pooled_methods"].values() if "encoder" in spec
    }
    for name in sorted(needed):
        member_paths = {"deterministic": None} if name == "raw_features" else paths[name]
        kind = None if name == "raw_features" else protocol["encoders"][name]["kind"]
        for member, path in member_paths.items():
            role_list = ("fit", *EVAL_ROLES, *REPLAY_ROLES)
            if kind is None:
                states = {role: raw_entity_features(samples[role]) for role in role_list}
                model = None
            else:
                model = load_encoder(kind, path, config, groups)
                encode = (encode_entities if kind == "graph" else encode_control_entities)
                states = {role: encode(model, samples[role], groups, batch_size=batch)
                          for role in role_list}
            for spec_name, entity_kind, how in entity_specs:
                if spec_name != name:
                    continue
                scorer = EntityScorer.fit(states["fit"], entity_kind,
                                          protocol["scoring"]["entity_estimator"])
                entity_scores = {role: scorer.entity_scores(states[role])
                                 for role in (*EVAL_ROLES, *REPLAY_ROLES)}
                graph = {role: aggregate(values, how) for role, values in entity_scores.items()}
                method = f"{name}|{entity_kind}|{how}"
                extra = None
                if entity_kind == "node":
                    extra = {"localization": localization_rate(
                        entity_scores["confirm_attack"],
                        [s.metadata["endpoint_keys"] for s in samples["confirm_attack"]],
                        [set(w["attack_endpoint_keys"]) for w in attack_info],
                    )}
                record(method, member, graph, extra)
                if method == protocol["protocol"]["primary"]:
                    replay[member] = auroc(graph["replay_test_normal"], graph["replay_test_attack"])
            for pooled, spec in protocol["pooled_methods"].items():
                if spec.get("encoder") != name:
                    continue
                embeddings = {role: embed_clean(model, samples[role], groups, batch_size=batch)
                              for role in ("fit", *EVAL_ROLES)}
                detector = MahalanobisDetector.fit(
                    embeddings["fit"], protocol["scoring"]["pooled_embedding_estimator"]
                )
                graph = {role: detector.score(embeddings[role]).numpy() for role in EVAL_ROLES}
                record(pooled, member, graph)
                pooled_thresholds.setdefault(pooled, {})[member] = conformal_threshold(
                    graph["calibration"], 0.95)
        print("done", name, flush=True)
    for pooled, spec in protocol["pooled_methods"].items():
        if "descriptors" not in spec:
            continue
        matrices = {role: descriptor_matrix(samples[role], spec["descriptors"], groups)
                    for role in ("fit", *EVAL_ROLES)}
        detector = MahalanobisDetector.fit(matrices["fit"], spec["estimator"], standardize=True)
        graph = {role: detector.score(matrices[role]).numpy() for role in EVAL_ROLES}
        record(pooled, "deterministic", graph)
        pooled_thresholds.setdefault(pooled, {})["deterministic"] = conformal_threshold(
            graph["calibration"], 0.95)

    # C2: pooled thresholds reproduce the frozen M1 geometry report.
    geometry = json.loads(
        (ROOT / "output/m1/xiiotid_stage1_m1/geometry_report.json").read_text()
    )
    c2 = {}
    for pooled, values in pooled_thresholds.items():
        name = pooled.split(":", 1)[1]
        for member, value in values.items():
            frozen = geometry["methods"][name]["members"][member]["calibration_thresholds"]["0.05"]
            c2[f"{pooled}#{member}"] = abs(value - frozen) / abs(frozen)
    c2_pass = max(c2.values()) <= 1e-9
    # C3: primary replayed on the consumed test reproduces the exploratory per-seed AUROC.
    exploration = json.loads(
        (ROOT / "output/diagnosis/entity_scoring_exploration.json").read_text()
    )
    explored = exploration["per_member"][protocol["protocol"]["primary"]]
    c3 = {member: abs(value - explored[member]["auroc"]) for member, value in replay.items()}
    c3_pass = max(c3.values()) <= 1e-9

    methods = list(members)
    primary = protocol["protocol"]["primary"]
    pairs = [(left, right) for left, right, _ in protocol["hypotheses"].values()]
    pairs += [(primary, m) for m in methods if m != primary and (primary, m) not in pairs]
    clusters = {
        "calibration": [s.metadata["session_key"] for s in samples["calibration"]],
        "normal": [s.metadata["session_key"] for s in samples["confirm_normal"]],
        "attack": [s.metadata["session_key"] for s in samples["confirm_attack"]],
    }
    bootstrap = cluster_bootstrap_detection(
        scores, members, clusters, pairs, targets,
        int(protocol["bootstrap"]["resamples"]), int(protocol["bootstrap"]["seed"]),
    )
    decisions = {}
    for name, (left, right, expected) in protocol["hypotheses"].items():
        interval = bootstrap["differences"][f"{left}__minus__{right}"]["auroc"]
        if expected == "greater":
            supported = interval["lower"] > 0
        elif expected == "less":
            supported = interval["upper"] < 0
        else:
            supported = interval["lower"] <= 0 <= interval["upper"]
        decisions[name] = {"expected": expected, "supported": bool(supported), **interval}

    def mean(method: str, getter) -> float:
        values = [getter(item) for item in per_member[method].values()]
        values = [v for v in values if v is not None and not math.isnan(float(v))]
        return float(np.mean(values)) if values else float("nan")

    def rate(metric: str, target: float):
        return lambda r: r["at_calibrated_fpr"][f"{target:g}"][metric]

    table = {
        method: {
            "auroc": mean(method, lambda r: r["auroc"]),
            "auroc_seed_std": float(np.std([r["auroc"] for r in per_member[method].values()])),
            "auprc": mean(method, lambda r: r["auprc"]),
            **{f"tpr@{t:g}": mean(method, rate("tpr", t)) for t in targets},
            **{f"fpr@{t:g}": mean(method, rate("fpr", t)) for t in targets},
            **{key: mean(method, lambda r, key=key: r[key])
               for key in ("size_stratified_auroc", "within_date_auroc", "auroc[fraction<0.1]",
                           "auroc[0.1<=fraction<0.5]", "auroc[fraction>=0.5]")},
            "localization_top1": mean(method, lambda r: r.get("localization", {}).get(
                "top1_is_attack_endpoint")),
            "members": len(per_member[method]),
        }
        for method in methods
    }
    calibration_check = table[primary]["fpr@0.05"] <= 0.10
    report = {
        "status": "complete",
        "protocol_sha256": registration["protocol_sha256"],
        "confirmation_opened_at_utc": opened_at,
        "counts": {role: len(samples[role]) for role in EVAL_ROLES},
        "dates": {
            "confirm_normal": len(set(strata_inputs["normal_dates"].tolist())),
            "confirm_attack": len(set(strata_inputs["attack_dates"].tolist())),
            "shared": len(set(strata_inputs["normal_dates"].tolist())
                          & set(strata_inputs["attack_dates"].tolist())),
        },
        "hypotheses": decisions,
        "calibration_check": {"primary_fpr@0.05": table[primary]["fpr@0.05"],
                              "pass": bool(calibration_check)},
        "consistency_checks": {
            "C1_audit_normal_reproduced": registration["C1_audit_normal_reproduced"],
            "C2_pooled_thresholds_match_m1_geometry": {"pass": c2_pass,
                                                       "max_relative_difference": max(c2.values())},
            "C3_primary_replay_matches_exploration": {"pass": c3_pass,
                                                      "max_absolute_difference": max(c3.values())},
        },
        "table": table,
        "per_member": per_member,
        "bootstrap": bootstrap,
        "finished_at_utc": now(),
    }
    report_path.write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        output / "confirmation_scores.npz",
        **{f"{k}#{role}": v for k, entry in scores.items() for role, v in entry.items()},
    )
    print(json.dumps({"hypotheses": {k: (v["supported"], round(v["mean_difference"], 3),
                                         round(v["lower"], 3), round(v["upper"], 3))
                                     for k, v in decisions.items()},
                      "calibration_check": report["calibration_check"],
                      "consistency": report["consistency_checks"]}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("register", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if args.phase == "register":
        register(protocol, args.protocol, args.output_dir)
    else:
        evaluate(protocol, args.protocol, args.output_dir)


if __name__ == "__main__":
    main()
