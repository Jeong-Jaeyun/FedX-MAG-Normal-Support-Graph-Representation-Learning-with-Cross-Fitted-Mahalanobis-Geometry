"""Entity-tail detector, trusted calibration, and poison-resistant causal state."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .contracts import GraphWindow, clean_batch, collate, require_normal
from .geometry import ConditionalMahalanobis
from .model import CausalEdgeGraphEncoder, EncoderConfig, EncoderOutput


def top_tail(values: Tensor, count: int) -> Tensor:
    """Fixed sparse-evidence aggregation; deliberately never averages all entities."""

    if values.ndim != 1 or not len(values):
        raise ValueError("Tail aggregation needs a nonempty one-dimensional score tensor")
    if count < 1:
        raise ValueError("tail count must be positive")
    return values.topk(min(count, len(values))).values.mean()


def conformal_threshold(scores: Iterable[float], fpr: float) -> float | None:
    """Finite-sample upper conformal threshold, or ``None`` if impossible."""

    values = np.sort(np.asarray(list(scores), dtype=np.float64))
    if not 0.0 < fpr < 1.0 or values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be finite and FPR must lie in (0, 1)")
    rank = math.ceil((len(values) + 1) * (1.0 - fpr))
    return float(values[rank - 1]) if rank <= len(values) else None


@dataclass
class CausalStateStore:
    """Keyed context that may be updated only after external normal verification."""

    hidden_dim: int
    nodes: dict[str, Tensor] = field(default_factory=dict)
    edges: dict[str, Tensor] = field(default_factory=dict)

    def read(self, window: GraphWindow, kind: str, device: torch.device) -> Tensor | None:
        keys = window.keys(kind)
        if keys is None:
            return None
        source = self.nodes if kind == "node" else self.edges
        values = [source.get(key, torch.zeros(self.hidden_dim)) for key in keys]
        return torch.stack(values).to(device=device, dtype=torch.float32)

    def admit_vetted_normal(self, window: GraphWindow, output: EncoderOutput) -> None:
        """Commit state only after normal status is externally established."""

        if window.label != 0:
            raise ValueError("Only externally vetted normal windows may update causal state")
        for kind, values, destination in (
            ("node", output.node_embedding, self.nodes),
            ("edge", output.edge_embedding, self.edges),
        ):
            keys = window.keys(kind)
            if keys is not None:
                for key, value in zip(keys, values, strict=True):
                    destination[key] = value.detach().cpu().clone()


@dataclass(frozen=True)
class DetectorConfig:
    tail_count: int = 3
    max_components: int = 8
    min_component_size: int = 32
    fpr_targets: tuple[float, ...] = (0.01, 0.05, 0.10)

    def __post_init__(self) -> None:
        if self.tail_count < 1 or self.max_components < 1 or self.min_component_size < 2:
            raise ValueError("invalid detector dimensions")
        if not self.fpr_targets or any(not 0.0 < value < 1.0 for value in self.fpr_targets):
            raise ValueError("FPR targets must lie in (0, 1)")


@dataclass(frozen=True)
class WindowScore:
    score: float
    node_tail: float
    edge_tail: float
    node_modes: Tensor
    edge_modes: Tensor
    node_energies: Tensor
    edge_energies: Tensor

    def as_prediction(
        self, window: GraphWindow, thresholds: dict[str, float | None]
    ) -> dict[str, Any]:
        def top(
            values: Tensor, modes: Tensor, keys: tuple[str, ...] | None
        ) -> list[dict[str, Any]]:
            order = torch.argsort(values, descending=True, stable=True)[: min(3, len(values))]
            return [
                {
                    "index": int(index),
                    "key": keys[int(index)] if keys is not None else None,
                    "energy": float(values[index]),
                    "normal_mode": int(modes[index]),
                }
                for index in order
            ]

        return {
            "session_key": window.metadata.get("session_key"),
            "window_start_ns": window.metadata.get("window_start_ns"),
            "score": self.score,
            "node_tail": self.node_tail,
            "edge_tail": self.edge_tail,
            "alerts": {
                name: None if threshold is None else bool(self.score > threshold)
                for name, threshold in thresholds.items()
            },
            "top_nodes": top(self.node_energies, self.node_modes, window.keys("node")),
            "top_edges": top(self.edge_energies, self.edge_modes, window.keys("edge")),
        }


@dataclass
class ConditionalSupportDetector:
    """Frozen encoder plus conditional normal support and separate calibration."""

    encoder_config: EncoderConfig
    encoder: CausalEdgeGraphEncoder
    node_support: ConditionalMahalanobis
    edge_support: ConditionalMahalanobis
    node_scale: float
    edge_scale: float
    calibration_scores: np.ndarray
    config: DetectorConfig = field(default_factory=DetectorConfig)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        encoder: CausalEdgeGraphEncoder,
        reference: Iterable[GraphWindow],
        selection: Iterable[GraphWindow],
        calibration: Iterable[GraphWindow],
        *,
        config: DetectorConfig | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> ConditionalSupportDetector:
        """Fit only on three disjoint vetted-normal roles.

        Callers own capture/site disjointness; the assertion below catches an
        accidental literal window overlap when IDs are available.
        """

        config = config or DetectorConfig()
        reference_items = require_normal(reference, "reference")
        selection_items = require_normal(selection, "selection")
        calibration_items = require_normal(calibration, "calibration")
        cls._assert_disjoint_windows(reference_items, selection_items, calibration_items)
        encoder.eval()
        reference_embedding = cls._embed_many(encoder, reference_items)
        node_support = ConditionalMahalanobis.fit(
            reference_embedding["node"],
            max_components=config.max_components,
            min_component_size=config.min_component_size,
        )
        edge_support = ConditionalMahalanobis.fit(
            reference_embedding["edge"],
            max_components=config.max_components,
            min_component_size=config.min_component_size,
            seed=31,
        )
        provisional = cls(
            encoder.config,
            encoder,
            node_support,
            edge_support,
            1.0,
            1.0,
            np.zeros(1, dtype=np.float64),
            config,
            dict(provenance or {}),
        )
        selected = provisional.score_many(selection_items)
        node_scale = max(float(np.quantile([item.node_tail for item in selected], 0.95)), 1e-9)
        edge_scale = max(float(np.quantile([item.edge_tail for item in selected], 0.95)), 1e-9)
        provisional.node_scale, provisional.edge_scale = node_scale, edge_scale
        provisional.calibration_scores = np.asarray(
            [item.score for item in provisional.score_many(calibration_items)], dtype=np.float64
        )
        # Do not silently claim a threshold that finite-sample calibration cannot provide.
        if conformal_threshold(provisional.calibration_scores, 0.05) is None:
            raise ValueError("At least 19 vetted calibration windows are needed for 5% FPR")
        return provisional

    @staticmethod
    def _window_id(window: GraphWindow) -> tuple[Any, Any] | None:
        session, timestamp = window.metadata.get("session_key"), window.timestamp()
        return (session, timestamp) if session is not None and timestamp is not None else None

    @classmethod
    def _assert_disjoint_windows(cls, *roles: list[GraphWindow]) -> None:
        indexed = [
            {item for item in (cls._window_id(window) for window in role) if item is not None}
            for role in roles
        ]
        for left in range(len(indexed)):
            for right in range(left + 1, len(indexed)):
                if indexed[left] & indexed[right]:
                    raise ValueError("reference, selection, and calibration windows overlap")

    @staticmethod
    @torch.no_grad()
    def _embed_one(
        encoder: CausalEdgeGraphEncoder,
        window: GraphWindow,
        state: CausalStateStore | None = None,
    ) -> EncoderOutput:
        batch = collate([window])
        device = next(encoder.parameters()).device
        batch = batch.to(device)
        node_memory = state.read(window, "node", device) if state else None
        edge_memory = state.read(window, "edge", device) if state else None
        return encoder(clean_batch(batch), node_memory=node_memory, edge_memory=edge_memory)

    @classmethod
    @torch.no_grad()
    def _embed_many(
        cls, encoder: CausalEdgeGraphEncoder, windows: Iterable[GraphWindow]
    ) -> dict[str, Tensor]:
        items = list(windows)
        embeddings = [cls._embed_one(encoder, item) for item in items]
        return {
            "node": torch.cat([item.node_embedding.detach().cpu() for item in embeddings]),
            "edge": torch.cat([item.edge_embedding.detach().cpu() for item in embeddings]),
        }

    def score_window(
        self, window: GraphWindow, state: CausalStateStore | None = None
    ) -> WindowScore:
        """Score without modifying temporal state or normal references."""

        output = self._embed_one(self.encoder, window, state)
        node_energies, node_modes = self.node_support.score(output.node_embedding.cpu())
        edge_energies, edge_modes = self.edge_support.score(output.edge_embedding.cpu())
        node_tail = float(top_tail(node_energies, self.config.tail_count))
        edge_tail = float(top_tail(edge_energies, self.config.tail_count))
        return WindowScore(
            max(node_tail / self.node_scale, edge_tail / self.edge_scale),
            node_tail,
            edge_tail,
            node_modes.cpu(),
            edge_modes.cpu(),
            node_energies.cpu(),
            edge_energies.cpu(),
        )

    def score_many(
        self, windows: Iterable[GraphWindow], state: CausalStateStore | None = None
    ) -> list[WindowScore]:
        return [self.score_window(window, state) for window in windows]

    @property
    def thresholds(self) -> dict[str, float | None]:
        return {
            f"{target:g}": conformal_threshold(self.calibration_scores, target)
            for target in self.config.fpr_targets
        }

    def admit_vetted_normal_to_context(self, window: GraphWindow, state: CausalStateStore) -> None:
        """Explicit post-verification state update; scoring itself is read-only."""

        output = self._embed_one(self.encoder, window, state)
        state.admit_vetted_normal(window, output)

    def prediction_rows(self, windows: Iterable[GraphWindow]) -> list[dict[str, Any]]:
        items = list(windows)
        return [
            score.as_prediction(window, self.thresholds)
            for window, score in zip(items, self.score_many(items), strict=True)
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "encoder_config": self.encoder_config.to_dict(),
            "encoder": {
                key: value.detach().cpu() for key, value in self.encoder.state_dict().items()
            },
            "node_support": self.node_support.state_dict(),
            "edge_support": self.edge_support.state_dict(),
            "node_scale": self.node_scale,
            "edge_scale": self.edge_scale,
            # Tensor-only persistence keeps ``torch.load(weights_only=True)``
            # usable for detector artifacts.
            "calibration_scores": torch.as_tensor(self.calibration_scores).cpu(),
            "detector_config": asdict(self.config),
            "provenance": self.provenance,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> ConditionalSupportDetector:
        if state.get("format_version") != 1:
            raise ValueError("Unsupported detector artifact")
        encoder_config = EncoderConfig(**state["encoder_config"])
        encoder = CausalEdgeGraphEncoder(encoder_config)
        encoder.load_state_dict(state["encoder"])
        encoder.eval()
        return cls(
            encoder_config,
            encoder,
            ConditionalMahalanobis.from_state_dict(state["node_support"]),
            ConditionalMahalanobis.from_state_dict(state["edge_support"]),
            float(state["node_scale"]),
            float(state["edge_scale"]),
            np.asarray(state["calibration_scores"], dtype=np.float64),
            DetectorConfig(**state["detector_config"]),
            dict(state.get("provenance", {})),
        )
