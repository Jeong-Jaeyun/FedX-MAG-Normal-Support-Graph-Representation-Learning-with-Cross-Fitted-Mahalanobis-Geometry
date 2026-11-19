"""Export address hashes solely as transient graph grouping keys, never features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .experiments import sha256, write_json
from .native_context import pack_file


def run(args: argparse.Namespace) -> None:
    source, output = Path(args.data).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with np.load(source) as payload:
        signature, times = payload["signature"], payload["observed_at"]
        dataset = str(payload.get("dataset", "ton"))
    if args.native_run:
        native = Path(args.native_run).resolve()
        registration = json.loads((native / "registration.json").read_text())
        native_input = registration.get("query_data_path") or registration["args"]["data"]
        with np.load(native_input) as original:
            if not np.array_equal(signature, original["signature"]):
                raise ValueError("Native events and query dataset rows differ")
        paths = [native / "events_sorted.npy"]
    else:
        if dataset != "xiiotid":
            raise ValueError("Use an existing native event preparation for this dataset")
        path = output / "events.npy"
        pack_file(
            (
                str(Path(args.root).resolve() / "data/X-IIoTID/X-IIoTID dataset.csv"),
                str(source),
                str(path),
            )
        )
        paths = [path]
    keys = {
        name: np.zeros(len(signature), dtype=dtype)
        for name, dtype in (("source", np.uint64), ("destination", np.uint64), ("port", np.int64))
    }
    seen = np.zeros(len(signature), dtype=bool)
    for path in paths:
        events = np.load(path, mmap_mode="r")
        observed = events[events["query"] >= 0]
        positions = observed["query"]
        if np.any(positions >= len(signature)) or not np.array_equal(
            observed["time"], times[positions]
        ):
            raise ValueError("Graph keys are not aligned to completed query observations")
        unique, first, inverse = np.unique(positions, return_index=True, return_inverse=True)
        for name in keys:
            if not np.array_equal(observed[name], observed[name][first][inverse]):
                raise ValueError("Aliased query observation has conflicting entity keys")
            keys[name][unique] = observed[name][first]
        seen[unique] = True
    if not seen.all():
        raise ValueError("Some observations lack graph grouping keys")
    np.savez_compressed(output / "keys.npz", signature=signature, **keys)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "data_sha256": sha256(source),
            "event_sha256": {p.name: sha256(p) for p in paths},
            "use": "group node incidence within a completed minute; never model input",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--native-run")
    parser.add_argument("--root", default=".")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
