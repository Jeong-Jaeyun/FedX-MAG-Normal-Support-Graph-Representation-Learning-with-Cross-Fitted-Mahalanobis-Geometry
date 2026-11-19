"""Diagnostic only: exact equal inputs shared by normal and attack rows.

Labels are used ONLY to describe representation ambiguity, never to fit a model.
This does not bound a temporal, node or window detector with additional inputs.
"""

import argparse
from pathlib import Path

import numpy as np

from fedxmag_next.experiments import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path, output = Path(args.data).resolve(), Path(args.output).resolve()
    with np.load(path) as payload:
        data = {key: payload[key] for key in payload.files}
    if str(data["dataset"]) != "ton" or data["features"].shape[1] != 109:
        raise ValueError("This diagnostic expects the full TON common+context schema")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {"args": vars(args), "input_sha256": sha256(path), "fit": "none; diagnostic only"},
    )
    features, labels, kinds = data["features"], data["label"], data["type"]
    times = data["observed_at"]
    offset = int(data["utc_offset_seconds"])
    cutoff = int(np.datetime64("2019-04-23", "s").astype("int64")) - offset
    end = int(np.datetime64("2019-05-01", "s").astype("int64")) - offset
    target = (times >= cutoff) & (times < end)
    results = {}
    for name, dimension in (("current", 73), ("full_history", 109)):
        rows = np.ascontiguousarray(features[:, :dimension])
        if not np.isfinite(rows).all():
            raise ValueError("Nonfinite feature values")
        rows[rows == 0] = 0  # Canonicalize the two floating zero representations.
        codes = rows.view(np.dtype((np.void, rows.dtype.itemsize * dimension))).ravel()
        normal_codes = np.unique(codes[target & (labels == 0)])
        collision = np.isin(codes, normal_codes)
        per_type = {}
        for category in np.unique(kinds[target & (labels == 1)]):
            chosen = target & (labels == 1) & (kinds == category)
            per_type[str(category)] = {
                "attack_rows": int(chosen.sum()),
                "exact_input_also_in_target_normal": int((chosen & collision).sum()),
                "fraction": float(collision[chosen].mean()),
            }
        results[name] = per_type
        print(name, per_type, flush=True)
    write_json(
        output / "report.json",
        {
            "status": "representation_diagnostic_only",
            "comparison": "target attack row vs any target normal row, exact numeric features",
            "limits": "Not a bound on temporal/node/window detectors; no classifier was trained",
            "results": results,
        },
    )


if __name__ == "__main__":
    main()
