"""Attach grouping-only entity keys to an old immutable node-feature cache."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .experiments import sha256, write_json


def run(args):
    output = Path(args.output).resolve()
    cache_path = Path(args.cache).resolve()
    data_path = Path(args.data).resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    if cache["data_sha256"] != sha256(data_path):
        raise ValueError("Cache observations differ from supplied data")
    with np.load(data_path) as data, np.load(args.keys) as keys:
        if not np.array_equal(data["signature"], keys["signature"]):
            raise ValueError("Entity keys refer to different observations")
        minutes = data["observed_at"] // 60
        compound = np.empty(2 * len(minutes), dtype=[("minute", "<i8"), ("node", "<u8")])
        compound["minute"] = np.tile(minutes, 2)
        compound["node"] = np.r_[keys["source"], keys["destination"]]
    unique = np.unique(compound)
    if not np.array_equal(unique["minute"], cache["node_minutes"].numpy()):
        raise ValueError("Reconstructed node ordering differs from original cache")
    if any(len(values) != len(unique) for values in cache["node_features"].values()):
        raise ValueError("Feature cache node count differs")
    node_keys = torch.from_numpy(unique["node"].copy().view(np.int64))
    if "node_keys" in cache and not torch.equal(cache["node_keys"], node_keys):
        raise ValueError("Existing cache keys differ")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "cache_sha256": sha256(cache_path),
            "data_sha256": cache["data_sha256"],
            "keys_sha256": sha256(Path(args.keys)),
            "change": "grouping keys only; feature/role/time tensors preserved",
        },
    )
    torch.save({**cache, "node_keys": node_keys}, output / "node_features.pt")
    restored = torch.load(output / "node_features.pt", weights_only=True)
    for name, features in cache["node_features"].items():
        if not torch.equal(features, restored["node_features"][name]):
            raise AssertionError("Adding metadata changed learned feature vectors")
    write_json(
        output / "report.json",
        {
            "status": "grouping_metadata_added",
            "nodes": len(unique),
            "feature_values_exact": True,
            "output_sha256": sha256(output / "node_features.pt"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "data", "keys", "output"):
        parser.add_argument(f"--{name}", required=True)
    run(parser.parse_args())
