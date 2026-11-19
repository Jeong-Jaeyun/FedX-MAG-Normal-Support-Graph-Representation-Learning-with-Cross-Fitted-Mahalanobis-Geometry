"""Score expanded observations with the unchanged original learned source model."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .daily_detector import DailyDeepEnergyDetector
from .experiments import sha256, write_json
from .source_only import physical_role_template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    path, artifact, output = map(Path, (args.data, args.detector, args.output))
    base = torch.load(artifact, weights_only=True)
    origin = json.loads((artifact.parent / "registration.json").read_text())
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    if tuple(data["feature_names"]) != tuple(base["feature_names"]):
        raise ValueError("Frozen expansion must preserve the feature schema")
    with np.load(origin["args"]["data"]) as z:
        old_signature = z["signature"]
    roles, policy = physical_role_template(data, str(artifact), 300)
    original_train = base["source_role_indices"]["train"].numpy()
    mapped_train = np.searchsorted(data["signature"], old_signature[original_train])
    reference = original_train
    maximum = origin["args"]["max_reference"]
    if len(reference) > maximum:
        reference = reference[np.argsort(old_signature[reference])[:maximum]]
    mapped_reference = np.searchsorted(data["signature"], old_signature[reference])
    detector = DailyDeepEnergyDetector(base)
    normalized = detector.normalize_features(torch.from_numpy(data["features"][mapped_reference]))
    if not torch.equal(normalized[: len(base["raw_reference"])], base["raw_reference"]):
        raise ValueError("Original raw prototype feature values changed during expansion")
    state = copy.deepcopy(base)
    state["source_data_sha256"] = sha256(path)
    state["source_role_indices"] = {
        "train": torch.from_numpy(mapped_train.copy()),
        "selection": torch.from_numpy(roles["selection"].copy()),
        "calibration": torch.from_numpy(roles["calibration"].copy()),
    }
    state["source_role_policy"] = {
        **policy,
        "training_membership": "exact old observations",
        "neural_and_raw_parameters": "unchanged original source",
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "args": {**vars(args), "history_gap": 300, "max_reference": maximum},
            "data_sha256": sha256(path),
            "base_encoder_sha256": sha256(artifact),
            "fit": (
                "NONE; frozen weights, units, supports, prototypes; expanded normal selection/cal"
            ),
            "original_reference_values_exact": True,
        },
    )
    torch.save(state, output / "source_detector.pt")
    write_json(
        output / "report.json",
        {"status": "frozen_source_expanded_queries", "original_reference_values_exact": True},
    )


if __name__ == "__main__":
    main()
