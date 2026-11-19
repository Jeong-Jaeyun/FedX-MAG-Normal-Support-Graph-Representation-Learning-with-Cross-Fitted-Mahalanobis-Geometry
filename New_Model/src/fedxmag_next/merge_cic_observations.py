"""Expand CIC query coverage while preserving all existing measured features.

Legacy HTTP summaries remain available only where they were actually extracted.
New rows have no invented packet-payload summaries. Acquisition is label blind.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .experiments import sha256, write_json
from .record_experiment import corruption_groups

URI_FIELDS = frozenset(
    (
        "log_http_uri_length",
        "uri_percent_fraction",
        "uri_punctuation_fraction",
        "available_http_uri",
    )
)


def merge_arrays(expanded, legacy):
    positions = np.searchsorted(expanded["signature"], legacy["signature"])
    if np.any(positions >= len(expanded["signature"])) or not np.array_equal(
        expanded["signature"][positions], legacy["signature"]
    ):
        raise ValueError("All old observations must survive coverage expansion")
    for field in ("timestamp", "observed_at", "label", "type", "capture"):
        if not np.array_equal(expanded[field][positions], legacy[field]):
            raise ValueError(f"Expanded observations changed old metadata: {field}")
    new_names = {str(name): i for i, name in enumerate(expanded["feature_names"])}
    old_names = [str(name) for name in legacy["feature_names"]]
    if not set(new_names) <= set(old_names):
        raise ValueError("Expanded non-payload schema differs from legacy data")
    features = np.zeros((len(expanded["signature"]), len(old_names)), dtype=np.float32)
    for i, name in enumerate(old_names):
        if name in new_names:
            features[:, i] = expanded["features"][:, new_names[name]]
            if name not in URI_FIELDS and not np.array_equal(
                features[positions, i], legacy["features"][:, i]
            ):
                raise ValueError(f"Coverage changed an old non-HTTP feature: {name}")
        elif not name.startswith("payload_"):
            raise ValueError(f"Missing observed feature: {name}")
    features[positions] = legacy["features"]
    http = np.zeros((len(features), legacy["http_features"].shape[1]), dtype=np.float32)
    http[positions] = legacy["http_features"]
    inspected = np.zeros(len(features), dtype=bool)
    inspected[positions] = True
    result = {
        **expanded,
        "features": features,
        "feature_names": legacy["feature_names"].copy(),
        "http_features": http,
        "http_feature_names": legacy["http_feature_names"].copy(),
        "http_summary_inspected": inspected,
    }
    if not np.array_equal(result["features"][positions], legacy["features"]):
        raise ValueError("Legacy full feature vectors were not preserved exactly")
    corruption_groups(result["feature_names"])
    return result, positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--legacy", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data, legacy, output = map(Path, (args.data, args.legacy, args.output))
    with np.load(data) as z:
        expanded = {k: z[k] for k in z.files}
    with np.load(legacy) as z:
        old = {k: z[k] for k in z.files}
    result, positions = merge_arrays(expanded, old)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "input_sha256": sha256(data),
            "legacy_sha256": sha256(legacy),
            "sampling": "all time-valid non-self raw connection rows, no label/type selection",
            "http_acquisition": (
                "legacy observations only; new payload summaries unavailable, never fabricated"
            ),
            "old_features_bit_exact": True,
            "old_metadata_identical": True,
        },
    )
    np.savez_compressed(output / "records.npz", **result)
    write_json(
        output / "report.json",
        {
            "records": len(result["signature"]),
            "legacy_records": len(positions),
            "new_records": len(result["signature"]) - len(positions),
            "dimensions": result["features"].shape[1],
            "output_sha256": sha256(output / "records.npz"),
        },
    )


if __name__ == "__main__":
    main()
