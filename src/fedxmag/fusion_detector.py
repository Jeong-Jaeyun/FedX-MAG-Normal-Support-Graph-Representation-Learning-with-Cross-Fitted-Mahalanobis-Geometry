"""Raw-node + flow-context kNN detector with vetted-normal environment commissioning.

Two channels keep evidence that frozen M0 node states lose: availability-aware
standardized v3b node features and per-edge flow context
(:mod:`fedxmag.flow_context`). An entity's score is its mean Euclidean distance
to the k nearest train-normal reference entities of its channel; a window's
channel score is the mean of its min(3, n) highest entity scores. Channel scores
are divided by their select-normal 95th percentile and fused by maximum.
Split-conformal thresholds come from a separate calibration-normal role. The flow
channel may use one of the fixed candidate variants in ``FLOW_VARIANTS``.

Commissioning a new environment appends independently vetted normal windows to
both references and recalibrates either on disjoint vetted normal windows or by
time-block cross-fitting over the commissioning windows themselves. Feature
scalers, k, aggregation, and fusion scales stay fixed. Queries never join a
reference. Labels are read only to refuse non-normal fit or commissioning data.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from fedxmag.flow_context import FLOW_DIM, FLOW_FEATURES, FLOW_VARIANTS
from fedxmag.local_support import LocalSupport
from fedxmag.m1.detection import Standardizer, conformal_threshold
from fedxmag.m1.entity import aggregate, raw_entity_features
from fedxmag.schema import FeatureGroups, GraphSample
from fedxmag.serialization import schema_fingerprint

METHOD = "raw_node_knn_plus_flow_knn_top3_select_q95_max"
# Version 2 adds the flow variant and calibration sessions; version 1 loads as the
# "all" variant with unknown calibration coverage.
FORMAT_VERSION = 2
CHANNELS = ("raw_node", "flow")
FPR_TARGETS = (0.01, 0.05, 0.10)
PRIMARY_FPR_TARGET = 0.05
DEFAULT_K = 20
MIN_SELECTION_WINDOWS = 20


@dataclass(frozen=True)
class TrafficWindows:
    """Graph windows (standardized with the detector's graph scaler) and their flow context."""

    graphs: Sequence[GraphSample]
    flows: Sequence[Tensor]

    def __post_init__(self) -> None:
        if len(self.graphs) != len(self.flows):
            raise ValueError("Each graph window needs exactly one flow-context matrix")
        for flows in self.flows:
            if flows.ndim != 2 or flows.shape[1] != FLOW_DIM or not torch.isfinite(flows).all():
                raise ValueError(f"Flow context must be a finite [edges, {FLOW_DIM}] matrix")

    def __len__(self) -> int:
        return len(self.graphs)

    def window_ids(self) -> set[tuple[Any, Any]]:
        return {
            (sample.metadata.get("session_key"), sample.metadata["window_start_ns"])
            for sample in self.graphs
            if "window_start_ns" in sample.metadata
        }

    def sessions(self) -> set[str]:
        return {
            sample.metadata["session_key"]
            for sample in self.graphs
            if sample.metadata.get("session_key") is not None
        }

    def subset(self, indices: Sequence[int]) -> TrafficWindows:
        return TrafficWindows([self.graphs[i] for i in indices], [self.flows[i] for i in indices])


def time_blocks(windows: TrafficWindows, count: int) -> list[list[int]]:
    """Split each capture chronologically into ``count`` contiguous blocks; pool block b."""

    sessions: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(windows.graphs):
        if sample.metadata.get("session_key") is None or "window_start_ns" not in sample.metadata:
            raise ValueError("Time blocks need session_key and window_start_ns metadata")
        sessions[sample.metadata["session_key"]].append(index)
    blocks: list[list[int]] = [[] for _ in range(count)]
    for members in sessions.values():
        ordered = sorted(members, key=lambda i: windows.graphs[i].metadata["window_start_ns"])
        for block, part in zip(blocks, np.array_split(np.asarray(ordered), count), strict=True):
            block.extend(part.tolist())
    return [sorted(block) for block in blocks]


def calibrated_thresholds(scores: Sequence[float]) -> dict[str, float | None]:
    """Conformal thresholds; ``None`` where calibration is too small for the target."""

    size = len(scores)
    return {
        f"{target:g}": conformal_threshold(scores, 1.0 - target)
        if math.ceil((size + 1) * (1.0 - target)) <= size
        else None
        for target in FPR_TARGETS
    }


def _check(role: str, windows: TrafficWindows, groups: FeatureGroups, normal_only: bool) -> None:
    if normal_only and (not len(windows) or any(s.label != 0 for s in windows.graphs)):
        raise ValueError(f"{role} must contain nonempty normal-only windows")
    for sample in windows.graphs:
        sample.validate(groups)


def _variant_rows(variant: str, flows: Sequence[Tensor]) -> list[Tensor]:
    """Apply a flow variant window by window (window-relative needs window boundaries)."""

    transform = FLOW_VARIANTS[variant]
    return [transform(matrix.double()) for matrix in flows]


def _knn(query: Tensor, reference: Tensor, k: int) -> Tensor:
    values = query.detach().cpu().double()
    if not len(values):
        return values.new_empty(0)
    distances, _ = LocalSupport.neighbours(values, reference, k)
    return distances.mean(dim=1)


@dataclass
class RawFlowDetector:
    groups: FeatureGroups
    node_reference: Tensor
    flow_reference: Tensor
    flow_scaler: Standardizer
    scales: dict[str, float]
    calibration_scores: np.ndarray
    k: int = DEFAULT_K
    environment: str = "source"
    reference_sessions: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    flow_variant: str = "all"
    calibration_sessions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.flow_variant not in FLOW_VARIANTS:
            raise ValueError(f"Unknown flow variant: {self.flow_variant}")
        expected = {"node": 2 * self.groups.node_dim, "flow": int(self.flow_scaler.keep.sum())}
        for kind, reference in (("node", self.node_reference), ("flow", self.flow_reference)):
            if reference.ndim != 2 or reference.shape[1] != expected[kind]:
                raise ValueError(f"The {kind} reference has the wrong feature dimension")
            if not torch.isfinite(reference).all():
                raise ValueError(f"The {kind} reference must be finite")
            if not 1 <= self.k < len(reference):
                raise ValueError("k must be positive and smaller than every reference size")
        if set(self.scales) != set(CHANNELS) or not all(
            math.isfinite(value) and value > 0 for value in self.scales.values()
        ):
            raise ValueError("Each channel needs a positive finite fusion scale")
        scores = np.asarray(self.calibration_scores, dtype=np.float64)
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise ValueError("Calibration requires a finite score vector")
        if calibrated_thresholds(scores)[f"{PRIMARY_FPR_TARGET:g}"] is None:
            raise ValueError("At least 19 calibration windows are needed for the 5% FPR target")
        self.calibration_scores = scores.copy()
        self.reference_sessions = tuple(sorted(self.reference_sessions))
        self.calibration_sessions = tuple(sorted(self.calibration_sessions))

    @classmethod
    def fit(
        cls,
        groups: FeatureGroups,
        train: TrafficWindows,
        selection: TrafficWindows,
        calibration: TrafficWindows,
        *,
        k: int = DEFAULT_K,
        flow_variant: str = "all",
        provenance: dict[str, Any] | None = None,
    ) -> RawFlowDetector:
        """Fit references on train, fusion scales on selection, thresholds on calibration."""

        for role, windows in (("train", train), ("selection", selection),
                              ("calibration", calibration)):
            _check(role, windows, groups, normal_only=True)
        if len(selection) < MIN_SELECTION_WINDOWS:
            raise ValueError("At least 20 selection windows are needed for the q95 fusion scales")
        if flow_variant not in FLOW_VARIANTS:
            raise ValueError(f"Unknown flow variant: {flow_variant}")
        flows = torch.cat(_variant_rows(flow_variant, train.flows))
        flow_scaler = Standardizer.fit(flows)
        # Unit scales and a placeholder calibration are replaced once references exist.
        detector = cls(
            groups,
            torch.cat(raw_entity_features(train.graphs).node),
            flow_scaler(flows),
            flow_scaler,
            dict.fromkeys(CHANNELS, 1.0),
            np.zeros(19),
            k,
            reference_sessions=tuple(train.sessions()),
            provenance=dict(provenance or {}),
            flow_variant=flow_variant,
        )
        _, channels, _ = detector.score(selection)
        detector = replace(
            detector,
            scales={
                name: max(float(np.quantile(channels[name], 0.95)), 1e-10) for name in CHANNELS
            },
        )
        return replace(
            detector,
            calibration_scores=detector.score(calibration)[0],
            calibration_sessions=tuple(calibration.sessions()),
        )

    def score(
        self, windows: TrafficWindows
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, list[Tensor]]]:
        """Fused window scores, unnormalized channel window scores, and entity scores."""

        _check("query", windows, self.groups, normal_only=False)
        rows = {
            "raw_node": raw_entity_features(windows.graphs).node,
            "flow": [
                self.flow_scaler(flows) for flows in _variant_rows(self.flow_variant, windows.flows)
            ],
        }
        references = {"raw_node": self.node_reference, "flow": self.flow_reference}
        entities, channels = {}, {}
        for name in CHANNELS:
            values = torch.cat(rows[name]) if rows[name] else references[name][:0]
            entities[name] = list(
                _knn(values, references[name], self.k).split([len(t) for t in rows[name]])
            )
            channels[name] = aggregate(entities[name], "top3_mean")
        fused = np.maximum(
            channels["raw_node"] / self.scales["raw_node"], channels["flow"] / self.scales["flow"]
        )
        if not np.isfinite(fused).all():
            raise ValueError("The detector produced nonfinite scores")
        return fused, channels, entities

    @property
    def thresholds(self) -> dict[str, float | None]:
        return calibrated_thresholds(self.calibration_scores)

    def recalibrate(self, calibration: TrafficWindows, *, environment: str) -> RawFlowDetector:
        """Threshold-only adaptation. It cannot repair ranking for unseen normal modes."""

        _check("calibration", calibration, self.groups, normal_only=True)
        history = [
            *self.provenance.get("commissioning", []),
            {
                "environment": environment,
                "action": "recalibrated",
                "calibration_windows": len(calibration),
                "calibration_sessions": sorted(calibration.sessions()),
            },
        ]
        return replace(
            self,
            calibration_scores=self.score(calibration)[0],
            calibration_sessions=tuple(calibration.sessions()),
            environment=environment,
            provenance={**self.provenance, "commissioning": history},
        )

    def _extended(self, reference: TrafficWindows) -> RawFlowDetector:
        """Append reference windows to both channels; calibration is left unchanged."""

        if not len(reference):
            return self
        return replace(
            self,
            node_reference=torch.cat(
                (self.node_reference, *raw_entity_features(reference.graphs).node)
            ),
            flow_reference=torch.cat(
                (
                    self.flow_reference,
                    self.flow_scaler(torch.cat(_variant_rows(self.flow_variant, reference.flows))),
                )
            ),
            reference_sessions=tuple(set(self.reference_sessions) | reference.sessions()),
        )

    def commission(
        self,
        reference: TrafficWindows,
        calibration: TrafficWindows | None = None,
        *,
        environment: str,
        blocks: int | None = None,
        record: dict[str, Any] | None = None,
    ) -> RawFlowDetector:
        """Add vetted normal reference windows, then recalibrate the commissioned detector.

        Give exactly one calibration source. ``calibration`` holds disjoint vetted normal
        windows. With ``blocks`` instead, the reference windows also calibrate: each
        capture is split chronologically into ``blocks`` contiguous blocks, and every
        block (pooled across captures) is scored against the source reference plus the
        other blocks, never against itself. The final reference holds every window.
        Contiguous blocks limit, but do not remove, optimism from temporal dependence at
        block edges; held-out references are smaller than the final one, which is
        conservative. Scalers and fusion scales are unchanged, and the original
        detector is not modified. ``record`` (for example input hashes) is kept in the
        commissioning history.
        """

        if (calibration is None) == (blocks is None):
            raise ValueError("Give either disjoint calibration windows or a time-block count")
        _check("commissioning reference", reference, self.groups, normal_only=True)
        step = {
            **(record or {}),
            "environment": environment,
            "action": "reference_added",
            "reference_windows": len(reference),
            "reference_sessions": sorted(reference.sessions()),
        }
        history = [*self.provenance.get("commissioning", []), step]
        if calibration is not None:
            _check("commissioning calibration", calibration, self.groups, normal_only=True)
            if reference.window_ids() & calibration.window_ids():
                raise ValueError("Commissioning reference and calibration windows must be disjoint")
            augmented = replace(
                self._extended(reference),
                provenance={**self.provenance, "commissioning": history},
            )
            return augmented.recalibrate(calibration, environment=environment)
        if blocks < 2:
            raise ValueError("Cross-fitted calibration needs at least two time blocks")
        scores = np.empty(len(reference))
        for members in time_blocks(reference, blocks):
            if members:
                held_out = set(members)
                others = [i for i in range(len(reference)) if i not in held_out]
                extended = self._extended(reference.subset(others))
                scores[members] = extended.score(reference.subset(members))[0]
        step.update(
            action="reference_added_cross_fitted_calibration",
            blocks=blocks,
            calibration_windows=len(reference),
        )
        return replace(
            self._extended(reference),
            calibration_scores=scores,
            calibration_sessions=tuple(reference.sessions()),
            environment=environment,
            provenance={**self.provenance, "commissioning": history},
        )

    def predict(
        self,
        windows: TrafficWindows,
        flow_keys: Sequence[Sequence[Sequence[str]]] | None = None,
    ) -> list[dict[str, Any]]:
        """Alerts and localization; labels and attack metadata are not used."""

        fused, channels, entities = self.score(windows)
        if flow_keys is not None and [len(keys) for keys in flow_keys] != [
            len(flows) for flows in windows.flows
        ]:
            raise ValueError("flow_keys must align with flow-context rows")
        thresholds = self.thresholds
        output = []
        for index, (sample, score) in enumerate(zip(windows.graphs, fused, strict=True)):
            endpoint_keys = sample.metadata.get("endpoint_keys")
            nodes, flows = entities["raw_node"][index], entities["flow"][index]
            if endpoint_keys is not None and len(endpoint_keys) != nodes.numel():
                raise ValueError("endpoint_keys must align with node rows")
            normalized = {
                name: float(channels[name][index] / self.scales[name]) for name in CHANNELS
            }
            session = sample.metadata.get("session_key")
            # None when capture identity or coverage is unknown. False flags a capture
            # whose normal traffic did not shape the reference or the thresholds.
            covered = {
                name: session in sessions if session is not None and sessions else None
                for name, sessions in (
                    ("reference", self.reference_sessions),
                    ("calibration", self.calibration_sessions),
                )
            }
            output.append(
                {
                    "index": index,
                    "session_key": session,
                    "window_start_ns": sample.metadata.get("window_start_ns"),
                    "environment": self.environment,
                    "capture_in_reference": covered["reference"],
                    "capture_in_calibration": covered["calibration"],
                    "score": float(score),
                    "channels": normalized,
                    "dominant_channel": max(CHANNELS, key=normalized.__getitem__),
                    "alerts": {
                        target: None if value is None else bool(score > value)
                        for target, value in thresholds.items()
                    },
                    # Stable ties keep localization reproducible.
                    "top_nodes": [
                        {
                            "node_index": item,
                            "endpoint_key": (
                                endpoint_keys[item] if endpoint_keys is not None else None
                            ),
                            "distance": float(nodes[item]),
                        }
                        for item in torch.argsort(nodes, descending=True, stable=True)[:3].tolist()
                    ],
                    "top_flows": [
                        {
                            "flow_index": item,
                            "edge_key": (
                                list(flow_keys[index][item]) if flow_keys is not None else None
                            ),
                            "distance": float(flows[item]),
                        }
                        for item in torch.argsort(flows, descending=True, stable=True)[:3].tolist()
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
            "flow_features": list(FLOW_FEATURES),
            "k": self.k,
            "node_reference": self.node_reference,
            "flow_reference": self.flow_reference,
            "flow_scaler": asdict(self.flow_scaler),
            "scales": dict(self.scales),
            "calibration_scores": torch.from_numpy(self.calibration_scores.copy()),
            "thresholds": self.thresholds,
            "environment": self.environment,
            "reference_sessions": list(self.reference_sessions),
            "provenance": self.provenance,
            "flow_variant": self.flow_variant,
            "calibration_sessions": list(self.calibration_sessions),
        }
        with Path(path).open("xb") as stream:
            torch.save(payload, stream)

    @classmethod
    def load(cls, path: str | Path) -> RawFlowDetector:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") not in (1, FORMAT_VERSION) or payload.get(
            "method"
        ) != METHOD:
            raise ValueError("Unsupported raw-flow detector format or scoring method")
        groups = FeatureGroups(**{key: tuple(value) for key, value in payload["groups"].items()})
        if schema_fingerprint(groups) != payload["schema_sha256"]:
            raise ValueError("Detector feature schema fingerprint differs")
        if tuple(payload["flow_features"]) != FLOW_FEATURES:
            raise ValueError("Detector flow-context features differ")
        detector = cls(
            groups,
            payload["node_reference"],
            payload["flow_reference"],
            Standardizer(**payload["flow_scaler"]),
            payload["scales"],
            payload["calibration_scores"].numpy(),
            payload["k"],
            payload["environment"],
            tuple(payload["reference_sessions"]),
            payload["provenance"],
            payload.get("flow_variant", "all"),
            tuple(payload.get("calibration_sessions", ())),
        )
        if detector.thresholds != payload["thresholds"]:
            raise ValueError("Stored thresholds differ from calibration scores")
        return detector
