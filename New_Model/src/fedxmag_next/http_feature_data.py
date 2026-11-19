"""Attach already-extracted, label-free HTTP summaries without changing rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .experiments import sha256, write_json
from .record_experiment import corruption_groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.data).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with np.load(source) as payload:
        data = {key: payload[key] for key in payload.files}
    if "http_features" not in data:
        raise ValueError("Expected a completed label-free HTTP extraction")
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "status": "same_rows_http_summary_extension",
            "input_sha256": sha256(source),
            "row_identity_and_labels_unchanged": True,
            "corruption": "URI and all derived HTTP summaries form one joint dependency group",
        },
    )
    data["features"] = np.concatenate((data["features"], data["http_features"]), axis=1)
    data["feature_names"] = np.concatenate(
        (
            data["feature_names"],
            np.asarray([f"payload_{name}" for name in data["http_feature_names"]]),
        )
    )
    corruption_groups(data["feature_names"])
    np.savez_compressed(output / "records.npz", **data)
    write_json(
        output / "report.json",
        {
            "records": len(data["features"]),
            "dimensions": data["features"].shape[1],
            "http_observed_records": int(np.sum(data["http_features"][:, 0] > 0)),
            "output_sha256": sha256(output / "records.npz"),
        },
    )


if __name__ == "__main__":
    main()
