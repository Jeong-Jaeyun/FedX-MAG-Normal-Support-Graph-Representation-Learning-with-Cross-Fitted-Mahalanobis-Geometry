"""Ablate historical scopes without changing observations or their provenance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .experiments import sha256, write_json
from .record_experiment import corruption_groups
from .temporal_context import CONTEXT_FEATURES, SCOPES

POLICIES = {
    "current": (),
    "pair": ("pair_service",),
    "outgoing_pair": ("outgoing", "pair_service"),
    "all": SCOPES,
}


def columns(names: np.ndarray, policy: str) -> np.ndarray:
    """Keep intrinsic features and the declared context scopes, in original order."""
    corruption_groups(names)
    context = {name for name in names if name.startswith("context_")}
    if context != set(CONTEXT_FEATURES):
        raise ValueError("Expected complete registered historical schema before ablation")
    prefixes = tuple(f"context_{scope}_" for scope in POLICIES[policy])
    return np.asarray(
        [
            index
            for index, name in enumerate(names)
            if not name.startswith("context_") or name.startswith(prefixes)
        ],
        dtype=np.int64,
    )


def run(args: argparse.Namespace) -> None:
    source, output = Path(args.data).resolve(), Path(args.output).resolve()
    with np.load(source) as payload:
        data = {key: payload[key] for key in payload.files}
    keep = columns(data["feature_names"], args.policy)
    if data["features"].shape[1] != len(data["feature_names"]):
        raise ValueError("Feature values and schema disagree")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "input_sha256": sha256(source),
            "selected_columns": keep.tolist(),
            "policy": "fixed feature names only; no row, time, label or type selection",
        },
    )
    data["features"] = data["features"][:, keep]
    data["feature_names"] = data["feature_names"][keep]
    corruption_groups(data["feature_names"])
    path = output / "records.npz"
    np.savez_compressed(path, **data)
    with np.load(path) as saved, np.load(source) as original:
        for key in original.files:
            expected = original[key]
            if key == "features":
                expected = expected[:, keep]
            elif key == "feature_names":
                expected = expected[keep]
            if not np.array_equal(saved[key], expected):
                raise AssertionError(f"Subset round-trip differs: {key}")
    write_json(
        output / "report.json",
        {
            "status": "label_independent_scope_ablation",
            "rows": len(data["features"]),
            "dimensions": data["features"].shape[1],
            "historical_scopes": list(POLICIES[args.policy]),
            "roundtrip_every_field_exact": True,
            "output_sha256": sha256(path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
