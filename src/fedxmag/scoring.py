"""Reusable frozen-M0 detector: node Mahalanobis energies and top-three mean.

Inputs are graphs already transformed by the encoder's train-fitted feature
standardizer. Fitting estimates only the normal entity distribution; encoder
parameters never receive gradients. Each seed is calibrated independently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fedxmag.m1.covariance import CovarianceEstimate
from fedxmag.m1.detection import MahalanobisDetector, conformal_threshold
from fedxmag.m1.entity import EntityScorer, StateGraphModel, aggregate, encode_entities
from fedxmag.schema import FeatureGroups, GraphSample
from fedxmag.serialization import schema_fingerprint

METHOD = "m0_graph_node_ledoit_wolf_top3_mean"
FORMAT_VERSION = 1
FPR_TARGETS = (0.01, 0.05, 0.10)
MIN_CALIBRATION_WINDOWS = 99


@dataclass
class FrozenNodeDetector:
    model: StateGraphModel
    model_kwargs: dict[str, Any]
    groups: FeatureGroups
    scorer: EntityScorer
    calibration_scores: np.ndarray
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.model.cpu().eval().requires_grad_(False)
        scores = np.asarray(self.calibration_scores, dtype=np.float64)
        if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
            raise ValueError("Calibration requires a nonempty finite score vector")
        if len(scores) < MIN_CALIBRATION_WINDOWS:
            raise ValueError("At least 99 calibration windows are needed for the 1% FPR target")
        self.calibration_scores = scores.copy()
        if self.scorer.kind != "node" or self.scorer.detector.standardizer is not None:
            raise ValueError("The confirmed detector uses unstandardized final node states")
        if self.scorer.detector.dimension != self.model_kwargs["hidden_dim"]:
            raise ValueError("Detector and encoder state dimensions differ")

    @classmethod
    def fit(
        cls,
        model: StateGraphModel,
        model_kwargs: dict[str, Any],
        groups: FeatureGroups,
        train_normal: Sequence[GraphSample],
        calibration_normal: Sequence[GraphSample],
        *,
        batch_size: int = 32,
        provenance: dict[str, Any] | None = None,
    ) -> FrozenNodeDetector:
        for role, samples in (("train", train_normal), ("calibration", calibration_normal)):
            if not samples or any(sample.label != 0 for sample in samples):
                raise ValueError(f"{role} must contain nonempty normal-only graphs")
            for sample in samples:
                sample.validate(groups)
        if len(calibration_normal) < MIN_CALIBRATION_WINDOWS:
            raise ValueError("At least 99 calibration windows are needed for the 1% FPR target")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        model.cpu().eval().requires_grad_(False)
        reference = encode_entities(model, train_normal, groups, batch_size)
        scorer = EntityScorer.fit(reference, "node", "ledoit_wolf")
        calibration = encode_entities(model, calibration_normal, groups, batch_size)
        return cls(
            model,
            dict(model_kwargs),
            groups,
            scorer,
            scorer.graph_scores(calibration, "top3_mean"),
            dict(provenance or {}),
        )

    @property
    def thresholds(self) -> dict[str, float]:
        return {
            f"{target:g}": conformal_threshold(self.calibration_scores, 1.0 - target)
            for target in FPR_TARGETS
        }

    def score(
        self, samples: Sequence[GraphSample], *, batch_size: int = 32
    ) -> tuple[np.ndarray, list[Tensor]]:
        """Return window scores and aligned node energies; labels are not used."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for sample in samples:
            sample.validate(self.groups)
        self.model.eval()
        states = encode_entities(self.model, samples, self.groups, batch_size)
        nodes = self.scorer.entity_scores(states)
        windows = aggregate(nodes, "top3_mean")
        if not np.isfinite(windows).all() or any(not torch.isfinite(s).all() for s in nodes):
            raise ValueError("Encoder or detector produced nonfinite scores")
        return windows, nodes

    def predict(
        self, samples: Sequence[GraphSample], *, batch_size: int = 32
    ) -> list[dict[str, Any]]:
        windows, entities = self.score(samples, batch_size=batch_size)
        thresholds = self.thresholds
        output = []
        aligned = zip(samples, windows, entities, strict=True)
        for index, (sample, score, nodes) in enumerate(aligned):
            keys = sample.metadata.get("endpoint_keys")
            if keys is not None and len(keys) != nodes.numel():
                raise ValueError("endpoint_keys must align with node rows")
            # Stable ties keep localization reproducible and the mean uses min(3, n).
            highest = torch.argsort(nodes, descending=True, stable=True)[:3].tolist()
            output.append(
                {
                    "index": index,
                    "session_key": sample.metadata.get("session_key"),
                    "window_start_ns": sample.metadata.get("window_start_ns"),
                    "node_count": nodes.numel(),
                    "score": float(score),
                    "alerts": {target: bool(score > value) for target, value in thresholds.items()},
                    "top_nodes": [
                        {
                            "node_index": item,
                            "endpoint_key": keys[item] if keys is not None else None,
                            "energy": float(nodes[item]),
                        }
                        for item in highest
                    ],
                }
            )
        return output

    def save(self, path: str | Path) -> None:
        payload = {
            "format_version": FORMAT_VERSION,
            "method": METHOD,
            "groups": asdict(self.groups),
            "schema_sha256": schema_fingerprint(self.groups),
            "model_kwargs": self.model_kwargs,
            "model": self.model.state_dict(),
            "covariance": asdict(self.scorer.detector.estimate),
            "calibration_scores": torch.from_numpy(self.calibration_scores.copy()),
            "thresholds": self.thresholds,
            "provenance": self.provenance,
        }
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path: str | Path) -> FrozenNodeDetector:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != FORMAT_VERSION or payload.get("method") != METHOD:
            raise ValueError("Unsupported frozen node detector format or scoring method")
        groups = FeatureGroups(**{key: tuple(value) for key, value in payload["groups"].items()})
        if schema_fingerprint(groups) != payload["schema_sha256"]:
            raise ValueError("Detector feature schema fingerprint differs")
        kwargs = payload["model_kwargs"]
        if kwargs["node_dim"] != groups.node_dim or kwargs["edge_dim"] != groups.edge_dim:
            raise ValueError("Encoder dimensions differ from the feature schema")
        model = StateGraphModel(**kwargs)
        model.load_state_dict(payload["model"])
        scorer = EntityScorer(
            MahalanobisDetector(CovarianceEstimate(**payload["covariance"])), "node"
        )
        detector = cls(
            model,
            kwargs,
            groups,
            scorer,
            payload["calibration_scores"].numpy(),
            payload["provenance"],
        )
        if detector.thresholds != payload["thresholds"]:
            raise ValueError("Stored thresholds differ from calibration scores")
        return detector
