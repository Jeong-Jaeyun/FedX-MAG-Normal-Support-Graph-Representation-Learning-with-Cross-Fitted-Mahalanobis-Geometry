"""Materialize the audit-band detection set with the frozen v3b preprocessing.

The audit band (60-70% of every capture date) has never been scored by any
detector. Its normal windows exist as ``audit_normal.pt``; its attack windows
were discarded by preparation and are rebuilt here from the raw CSV with the
unchanged v3b code and the train-normal standardizer. As a pipeline check, the
band's normal windows are rebuilt too and must equal the stored
``audit_normal.pt`` exactly. No detector score is computed by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from fedxmag.config import (
    config_fingerprint,
    feature_groups_from_config,
    load_stage1_config,
    preprocessing_code_fingerprint,
)
from fedxmag.preprocessing import (
    FeatureStandardizer,
    build_static_graphs,
    chronological_window_aligned_split,
    read_xiiotid_events,
)
from fedxmag.serialization import load_graph_samples, save_graph_samples

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "processed" / "xiiotid_stage1_v3b_development"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_graph(left, right) -> bool:
    return (
        torch.equal(left.node_features, right.node_features)
        and torch.equal(left.edge_features, right.edge_features)
        and torch.equal(left.edge_index, right.edge_index)
        and torch.equal(left.node_available, right.node_available)
        and torch.equal(left.edge_available, right.edge_available)
        and left.label == right.label
        and left.metadata == right.metadata
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=ROOT / "data" / "X-IIoTID" / "X-IIoTID dataset.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data" / "processed" / "xiiotid_stage1_v3b_audit_band")
    args = parser.parse_args()
    config = load_stage1_config(ROOT / "configs" / "stage1_m0_v3_development.yaml")
    manifest = json.loads((PREPARED / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_sha256"] != config_fingerprint(config):
        raise ValueError("The v3b configuration differs from the prepared artifact")
    if manifest["preprocessing_code_sha256"] != preprocessing_code_fingerprint():
        raise ValueError("The preprocessing code differs from the prepared artifact")
    groups = feature_groups_from_config(config)
    data = config["data"]
    output = args.output_dir / "audit_attack.pt"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    events = read_xiiotid_events(args.input, deduplicate_exact_rows=bool(data["deduplicate_exact_rows"]))
    splits = chronological_window_aligned_split(
        events.events,
        {key: float(value) for key, value in data["split_fractions"].items()},
        window_seconds=int(data["window_seconds"]),
        session_split_strategy=str(data["session_split_strategy"]),
    )
    audit_events = splits["audit"]
    graphs = build_static_graphs(
        audit_events,
        window_seconds=int(data["window_seconds"]),
        groups=groups,
        timing_profile=str(config["features"]["timing_profile"]),
        timestamp_resolution_seconds=int(config["features"]["timestamp_resolution_seconds"]),
    )
    scaler = FeatureStandardizer.from_state_dict(
        torch.load(PREPARED / "standardizer.pt", map_location="cpu", weights_only=True)
    )
    normal = scaler.transform_many([g for g in graphs if g.label == 0], groups)
    attack = scaler.transform_many([g for g in graphs if g.label == 1], groups)

    stored = load_graph_samples(PREPARED / "audit_normal.pt", groups)
    if len(stored) != len(normal) or not all(
        same_graph(a, b) for a, b in zip(stored, normal, strict=True)
    ):
        raise AssertionError("Rebuilt audit-band normal windows differ from audit_normal.pt")

    window_ns = int(data["window_seconds"]) * 1_000_000_000
    records: dict[tuple[str, int], dict[str, object]] = {}
    for event in audit_events:
        if not event.timing_observed or event.src_key == event.dst_key:
            continue
        key = (event.session_key, (event.timestamp_ns // window_ns) * window_ns)
        entry = records.setdefault(key, {"records": 0, "attack_records": 0, "endpoints": set(),
                                         "pairs": set()})
        entry["records"] += 1
        if event.is_attack:
            entry["attack_records"] += 1
            entry["endpoints"].update((event.src_key, event.dst_key))
            entry["pairs"].add((event.src_key, event.dst_key))
    sidecar = []
    for sample in attack:
        entry = records[(sample.metadata["session_key"], sample.metadata["window_start_ns"])]
        sidecar.append({
            "session": sample.metadata["session_key"],
            "window_start_ns": sample.metadata["window_start_ns"],
            "records": entry["records"],
            "attack_records": entry["attack_records"],
            "attack_fraction": entry["attack_records"] / entry["records"],
            "attack_endpoint_keys": sorted(entry["endpoints"]),
            "attack_pair_keys": sorted(entry["pairs"]),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_graph_samples(output, attack, groups=groups)
    (args.output_dir / "audit_attack_windows.json").write_text(json.dumps(sidecar))
    built = {
        "status": "built",
        "purpose": "untouched audit-band detection confirmation set; no detector score computed",
        "audit_normal_rebuilt_identical": True,
        "counts": {"audit_normal": len(normal), "audit_attack": len(attack)},
        "sessions": {
            "audit_normal": len({s.metadata["session_key"] for s in normal}),
            "audit_attack": len({s.metadata["session_key"] for s in attack}),
        },
        "sha256": {
            "audit_attack.pt": sha256(output),
            "audit_attack_windows.json": sha256(args.output_dir / "audit_attack_windows.json"),
            "audit_normal.pt": sha256(PREPARED / "audit_normal.pt"),
            "standardizer.pt": sha256(PREPARED / "standardizer.pt"),
            "build_script": sha256(Path(__file__)),
        },
        "config_sha256": config_fingerprint(config),
        "preprocessing_code_sha256": preprocessing_code_fingerprint(),
        "raw_input_sha256": sha256(args.input),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(built, indent=2))
    print(json.dumps({k: built[k] for k in ("counts", "sessions", "audit_normal_rebuilt_identical")}))


if __name__ == "__main__":
    main()
