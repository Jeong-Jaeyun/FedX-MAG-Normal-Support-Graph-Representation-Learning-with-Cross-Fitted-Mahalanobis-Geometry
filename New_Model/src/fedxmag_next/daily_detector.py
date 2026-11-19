"""Read-only scoring from a daywise deep-energy artifact.

This restores all learned weights, support geometries, scalers, and fusion
constants without training or access to a query label. Calibration/feedback is
an explicit, separate chronological operation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .contrastive_energy import DependencyEnergy
from .experiments import knn_scores, write_json
from .geometry import ConditionalMahalanobis
from .geometry_training import GeometryResidualEncoder
from .neural_transforms import TransformationEnergy


def support_energy(support: ConditionalMahalanobis, values: Tensor) -> Tensor:
    """Bound triangular-solve memory independently of the complete query count."""
    return torch.cat([support.score(chunk)[0] for chunk in values.split(4096)])


class DailyDeepEnergyDetector:
    def __init__(self, state: dict) -> None:
        if state.get("format_version") not in {1, 2}:
            raise ValueError("Unsupported daywise detector format")
        self.state = state
        self.feature_names = tuple(state["feature_names"])
        self.encoder_kind = state.get("encoder_kind", "dependency")
        if self.encoder_kind not in (
            "dependency",
            "neural_transformations",
            "association",
            "geometry_margin",
            "frozen_residual",
        ):
            raise ValueError("Unsupported source encoder kind")
        self.models = []
        for item in state["seed_models"]:
            if self.encoder_kind == "neural_transformations":
                model = TransformationEnergy(
                    len(self.feature_names), **state["transformation_config"]
                )
            elif self.encoder_kind in ("geometry_margin", "frozen_residual"):
                model = GeometryResidualEncoder(len(self.feature_names))
            else:
                model = DependencyEnergy(len(self.feature_names))
            model.load_state_dict(item["model"])
            model.eval()
            self.models.append(model)

    @classmethod
    def load(cls, path: str | Path) -> DailyDeepEnergyDetector:
        return cls(torch.load(Path(path), map_location="cpu", weights_only=True))

    def normalize_features(self, features: Tensor) -> Tensor:
        values = (features.cpu().float() - self.state["input_mean"]) / self.state["input_scale"]
        keep = self.state.get("input_keep")
        return values if keep is None else values * keep

    @torch.no_grad()
    def score(self, features: Tensor, modes: int = 8) -> dict[str, Tensor]:
        return self.score_all(features, modes=(modes,))[modes]

    @torch.no_grad()
    def upgrade_knn_precision(self, selection_features: Tensor) -> None:
        """Upgrade a legacy artifact using held-out normal selection rows only."""
        values = self.normalize_features(selection_features)
        raw = knn_scores(self.state["raw_reference"], values, dtype=torch.float64)
        scale = max(float(torch.quantile(raw, 0.95)), 1e-6)
        for fusion in self.state["fusion_scales"].values():
            fusion["raw"] = scale
        self.state["format_version"] = 2
        self.state["knn_dtype"] = "float64"

    @torch.no_grad()
    def score_all(
        self, features: Tensor, modes: tuple[int, ...] = (1, 8)
    ) -> dict[int, dict[str, Tensor]]:
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("Features do not match the frozen detector schema")
        if not torch.isfinite(features).all():
            raise ValueError("Features must be finite; availability is part of the schema")
        values = self.normalize_features(features)
        embedded, native = [], []
        for model, seed_state in zip(self.models, self.state["seed_models"], strict=True):
            embedding = model.layers[:-1](values)
            embedding = (embedding - seed_state["embedding_mean"]) / seed_state["embedding_scale"]
            embedded.append(embedding)
            if self.encoder_kind in ("neural_transformations", "association"):
                energy = (
                    torch.cat([model(x) for x in values.split(8192)]).double()
                    if len(values)
                    else torch.empty(0, dtype=torch.float64)
                )
                native.append(
                    (energy - seed_state["native_score_center"]) / seed_state["native_score_scale"]
                    + 1
                )
        dtype = torch.float64 if self.state.get("knn_dtype") == "float64" else torch.float32
        raw = knn_scores(self.state["raw_reference"], values, dtype=dtype)
        results = {}
        for count in modes:
            neural = []
            for embedding, seed_state in zip(embedded, self.state["seed_models"], strict=True):
                support_state = seed_state["supports"][str(count)]
                support = ConditionalMahalanobis.from_state_dict(support_state["geometry"])
                energy = support_energy(support, embedding)
                neural.append(
                    (energy - support_state["score_center"]) / support_state["score_scale"] + 1
                )
            deep = torch.stack(neural).mean(dim=0)
            raw_support = ConditionalMahalanobis.from_state_dict(
                self.state["raw_supports"][f"raw_mahalanobis_m{count}"]
            )
            scales = self.state["fusion_scales"][str(count)]
            results[count] = {
                "raw_knn": raw,
                "raw_mahalanobis": support_energy(raw_support, values),
                "deep_mahalanobis": deep,
                "hybrid": torch.maximum(raw / scales["raw"], deep / scales["deep"]),
            }
            if native:
                key = "association" if self.encoder_kind == "association" else "transformation"
                results[count][key] = torch.stack(native).mean(0)
        return results


def audit(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    source, output = Path(args.run).resolve(), Path(args.output).resolve()
    with np.load(args.data) as payload:
        data = {key: payload[key] for key in payload.files}
    stored = torch.load(source / "scores.pt", weights_only=True)
    indices = stored["evaluation_indices"].numpy()
    report = {"status": "passed", "days": {}}
    for capture in sorted(set(data["capture"][indices])):
        chosen = np.flatnonzero(data["capture"][indices] == capture)
        detector = DailyDeepEnergyDetector.load(source / f"day_{capture}_detector.pt")
        if detector.feature_names != tuple(data["feature_names"]):
            raise ValueError("Audit input schema differs from detector")
        if detector.state["latest_reference_observed_at"] >= int(
            data["observed_at"][indices[chosen]].min()
        ):
            raise ValueError("Restored detector contains a future reference")
        results = {}
        for modes in (1, 8):
            scores = detector.score(torch.from_numpy(data["features"][indices[chosen]]), modes)
            pairs = {
                "raw_knn": "raw_knn",
                "deep_mahalanobis": f"deep_mahalanobis_m{modes}_ensemble",
                "hybrid": f"raw_plus_deep_m{modes}",
            }
            for key, method in pairs.items():
                expected = stored["scores"][method][chosen]
                actual = scores[key]
                relative = (actual - expected).abs() / (1 + expected.abs())
                if float(relative.max()) > 1e-5:
                    raise ValueError(f"Score replay differs for day {capture}, {method}")
                results[f"{modes}_{key}"] = {
                    "max_absolute_difference": float((actual - expected).abs().max()),
                    "max_scaled_difference": float(relative.max()),
                }
        report["days"][str(int(capture))] = results
    write_json(output, report)
    print(json.dumps({"audit": report["status"], "days": len(report["days"])}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
