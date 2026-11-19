"""Isolate a raw-feature bypass without retraining the existing dependency DNN."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .daily_detector import DailyDeepEnergyDetector, support_energy
from .experiments import sha256, write_json
from .geometry import ConditionalMahalanobis
from .geometry_training import GeometryResidualEncoder


@torch.no_grad()
def add_raw_bypass(state, reference_features, selection_features):
    base = DailyDeepEnergyDetector(state)
    if base.encoder_kind != "dependency":
        raise ValueError("This frozen ablation requires the original dependency encoder")
    result = copy.deepcopy(state)
    result["encoder_training_kind"] = "dependency_frozen_weights_no_neural_refit"
    result["encoder_kind"] = "frozen_residual"
    reference = base.normalize_features(reference_features)
    selected = base.normalize_features(selection_features)
    if not torch.equal(reference[: len(state["raw_reference"])], state["raw_reference"]):
        raise ValueError("Frozen bypass reference differs from its original capped fit rows")
    dimension = reference.shape[1]
    selection_scores = {1: [], 8: []}
    for original, item in zip(base.models, result["seed_models"], strict=True):
        model = GeometryResidualEncoder(dimension)
        model.layers[0].network.load_state_dict(original.layers[:-1].state_dict())
        model.eval()
        item["model"] = model.state_dict()
        item["embedding_mean"] = torch.cat((torch.zeros(dimension), item["embedding_mean"]))
        item["embedding_scale"] = torch.cat((torch.ones(dimension), item["embedding_scale"]))
        embedded = (model(reference) - item["embedding_mean"]) / item["embedding_scale"]
        embedded_selection = (model(selected) - item["embedding_mean"]) / item["embedding_scale"]
        item["supports"] = {}
        for modes in (1, 8):
            geometry = ConditionalMahalanobis.fit(embedded, max_components=modes)
            energy = support_energy(geometry, embedded_selection)
            center, high = torch.quantile(energy, torch.tensor([0.5, 0.95]).double())
            scale = max(float(high - center), 1e-6)
            item["supports"][str(modes)] = {
                "geometry": geometry.state_dict(),
                "score_center": float(center),
                "score_scale": scale,
            }
            selection_scores[modes].append((energy - center) / scale + 1)
        print("refit frozen residual geometry seed", item["seed"], flush=True)
    for modes in (1, 8):
        score = torch.stack(selection_scores[modes]).mean(0)
        result["fusion_scales"][str(modes)]["deep"] = max(float(torch.quantile(score, 0.95)), 1e-6)
    return result


def run(args):
    torch.set_num_threads(args.workers)
    path, artifact, output = map(Path, (args.data, args.detector, args.output))
    state = torch.load(artifact, map_location="cpu", weights_only=True)
    if sha256(path) != state["source_data_sha256"]:
        raise ValueError("Frozen bypass data differ from original encoder data")
    origin = json.loads((artifact.parent / "registration.json").read_text())
    with np.load(path) as payload:
        data = {key: payload[key] for key in ("features", "feature_names", "label", "signature")}
    if tuple(data["feature_names"]) != tuple(state["feature_names"]):
        raise ValueError("Frozen bypass feature schema differs")
    roles = state["source_role_indices"]
    reference = roles["train"].numpy()
    maximum = origin["args"]["max_reference"]
    if len(reference) > maximum:
        reference = reference[np.argsort(data["signature"][reference])[:maximum]]
    selection = roles["selection"].numpy()
    if np.any(data["label"][reference] != 0) or np.any(data["label"][selection] != 0):
        raise ValueError("Frozen residual geometry received attacks")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "args": {
                **vars(args),
                "max_reference": maximum,
                "history_gap": origin["args"]["history_gap"],
            },
            "data_sha256": sha256(path),
            "base_encoder_sha256": sha256(artifact),
            "fit": (
                "NO neural training; exact dependency hidden weights; raw+hidden support refit"
            ),
            "source_membership": "exact base role indices and original signature-capped reference",
            "raw_controls": "all original raw supports/references/scales preserved",
        },
    )
    updated = add_raw_bypass(
        state,
        torch.from_numpy(data["features"][reference]),
        torch.from_numpy(data["features"][selection]),
    )
    torch.save(updated, output / "source_detector.pt")
    write_json(output / "report.json", {"status": "frozen_dependency_raw_bypass_refit_complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
