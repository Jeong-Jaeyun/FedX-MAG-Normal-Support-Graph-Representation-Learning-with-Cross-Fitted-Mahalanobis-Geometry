"""Window prediction from an exported initial candidate, with explicit feedback.

Inputs are already completed-flow features in the packet's ordered schema.
This does not implement a live network capture or verify normality externally.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .causal_risk import CausalPeakMemory, scan_contributors
from .daily_detector import DailyDeepEnergyDetector, support_energy
from .detector import conformal_threshold
from .geometry import ConditionalMahalanobis
from .prototype_guard import PrototypeProjector, VerifiedPrototypeMemory, entity_keys
from .service_risk_keys import PROTOCOL_COLUMNS, service_keys
from .weak_evidence import CausalEvidenceMemory
from .window_bridge import aggregate_connections


class CandidateStream:
    """Predict one complete window, then commit external normal verification."""

    def __init__(self, packet):
        if packet.get("format_version") not in (1, 2, 3):
            raise ValueError("Unsupported candidate packet")
        if packet["path"] not in ("prototype", "prototype_risk_fixed"):
            raise ValueError("Unsupported exported candidate path")
        self.packet = packet
        model = packet["prototype_model"]
        self.base = DailyDeepEnergyDetector(model["base_encoder"])
        self.projector = PrototypeProjector(model["base_encoder"])
        self.normal_view = packet.get(
            "normal_metric", "deep" if packet["base_head"].startswith("deep_") else "raw"
        )
        self.normal = VerifiedPrototypeMemory(
            {self.normal_view: model["normal_reference"][self.normal_view]},
            {k: v.numpy().view(np.uint64) for k, v in model["normal_reference_keys"].items()},
            model["normal_reference_times"].numpy(),
            model["radii"],
            model["maximum_prototypes"],
        )
        reference = packet["row_reference"].numpy()
        self.reference = np.sort(reference)
        self.q95 = float(np.quantile(reference, 0.95))
        if self.q95 <= 0:
            raise ValueError("Candidate requires a positive source-normal row reference")
        self.evidence_rule = packet.get("evidence_rule")
        if self.evidence_rule and self.evidence_rule["initiation"] not in ("gated", "weak"):
            raise ValueError("Unsupported original-evidence initiation rule")
        self.risk = (
            CausalEvidenceMemory(self.q95, self.evidence_rule["mode"], **packet["risk_parameters"])
            if self.evidence_rule
            else CausalPeakMemory(self.q95, **packet["risk_parameters"])
            if packet["path"] != "prototype"
            else None
        )
        self.modes = "8" if "_m8" in packet["base_head"] else "1"
        if packet["base_head"].startswith("deep_"):
            self.supports = [
                ConditionalMahalanobis.from_state_dict(item["supports"][self.modes]["geometry"])
                for item in self.base.state["seed_models"]
            ]
        else:
            self.supports = [
                ConditionalMahalanobis.from_state_dict(
                    self.base.state["raw_supports"][packet["base_head"]]
                )
            ]
        initial = packet["source_window_calibration"].tolist()[-packet["capacity"] :]
        self.base_history = deque(initial, maxlen=packet["capacity"])
        self.final_history = deque(initial, maxlen=packet["capacity"])
        self.service = "destination_port" in packet.get("risk_key_fields", [])
        if self.service:
            self.service_columns = [
                self.base.feature_names.index(k) for k in (*PROTOCOL_COLUMNS, "available_dst_port")
            ]
        self.last_minute, self.pending = None, None

    @torch.no_grad()
    def row_energy(self, features):
        if features.ndim != 2 or features.shape[1] != len(self.base.feature_names):
            raise ValueError("Candidate features do not match its ordered schema")
        if not torch.isfinite(features).all():
            raise ValueError("Candidate input features must be finite")
        if not len(features):
            return np.empty(0, dtype=np.float64)
        normalized = self.base.normalize_features(features)
        if not self.packet["base_head"].startswith("deep_"):
            return support_energy(self.supports[0], normalized).numpy()
        values = []
        for network, item, support in zip(
            self.base.models, self.base.state["seed_models"], self.supports, strict=True
        ):
            hidden = network.layers[:-1](normalized)
            hidden = (hidden - item["embedding_mean"]) / item["embedding_scale"]
            config = item["supports"][self.modes]
            values.append(
                (support_energy(support, hidden) - config["score_center"]) / config["score_scale"]
                + 1
            )
        return torch.stack(values).mean(0).numpy()

    def aggregate(self, values):
        return (
            aggregate_connections(
                values, self.reference, self.packet["aggregation"], reference_sorted=True
            )
            if len(values)
            else 0.0
        )

    @torch.no_grad()
    def query(self, minute, features, keys, observed_at):
        if self.pending is not None or (
            self.last_minute is not None and minute <= self.last_minute
        ):
            raise ValueError("Commit feedback before a strictly later candidate query")
        times = np.asarray(observed_at, dtype=np.int64)
        if times.shape != (len(features),) or np.any(times // 60 != minute):
            raise ValueError("Every candidate observation must belong to its completed window")
        endpoint_keys = {k: np.asarray(keys[k]) for k in ("source", "destination")}
        if any(v.shape != times.shape for v in endpoint_keys.values()):
            raise ValueError("Candidate endpoints and observations must align")
        # Validate all metadata before changing either online state.
        active = entity_keys(endpoint_keys, self.packet["scope"])
        if self.service:
            metadata = features[:, self.service_columns].numpy()
            active = service_keys(keys, metadata[:, :4], metadata[:, 4], self.packet["scope"])
        raw = self.row_energy(features)
        projected = self.projector.project(features)
        now = (int(minute) + 1) * 60
        ratios = self.normal.query(
            now, {self.normal_view: projected[self.normal_view]}, endpoint_keys, times
        )[self.packet["scope"]][self.normal_view].numpy()
        filtered = raw.copy()
        finite = np.isfinite(ratios)
        filtered[finite] = np.minimum(filtered[finite], ratios[finite] ** 2 * self.q95)
        base_energy = self.aggregate(raw)
        base_threshold = conformal_threshold(self.base_history, self.packet["nominal_fpr"])
        predicted = filtered
        if self.risk is not None:
            predicted = self.risk.predict(now, active, filtered, times)
            familiar = ratios <= 1 + 1e-6
            predicted[familiar] = filtered[familiar]
            contributors = (
                scan_contributors(
                    filtered, self.reference, self.packet["aggregation"], reference_sorted=True
                )
                if len(filtered)
                and base_threshold is not None
                and self.aggregate(filtered) > base_threshold
                else np.empty(0, dtype=np.int64)
            )
            contributors = contributors[~familiar[contributors]]
            if self.evidence_rule:
                if self.evidence_rule["initiation"] == "weak":
                    contributors = np.flatnonzero((~familiar) & (filtered > self.q95))
                self.risk.commit(contributors)
            else:
                self.risk.feedback(verified_normal=False, contributors=contributors)
        energy = self.aggregate(predicted)
        threshold = conformal_threshold(self.final_history, self.packet["nominal_fpr"])
        valid = bool(len(features) and threshold is not None)
        self.pending = (int(minute), base_energy, energy, bool(len(features)))
        return {
            "energy": energy,
            "base_energy": base_energy,
            "threshold": threshold,
            "valid": valid,
            "alarm": bool(valid and energy > threshold),
        }

    def feedback(self, *, verified_normal):
        if self.pending is None or not isinstance(verified_normal, bool):
            raise ValueError("External boolean normal verification must follow a prediction")
        minute, base, energy, observed = self.pending
        self.normal.feedback(verified_normal=verified_normal)
        if verified_normal and observed:
            self.base_history.append(base)
            self.final_history.append(
                energy if self.packet["calibration_feedback"] == "candidate" else base
            )
        self.last_minute, self.pending = minute, None

    def state_dict(self):
        if self.pending is not None:
            raise ValueError("Checkpoint a candidate only after feedback is committed")
        return {
            "format_version": 1,
            "packet": self.packet,
            "normal": self.normal.state_dict(),
            "risk": self.risk.state_dict() if self.risk is not None else None,
            "base_history": list(self.base_history),
            "final_history": list(self.final_history),
            "last_minute": self.last_minute,
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("format_version") != 1:
            raise ValueError("Unsupported candidate stream checkpoint")
        instance = cls(state["packet"])
        instance.normal = VerifiedPrototypeMemory.from_state_dict(state["normal"])
        instance.risk = (
            (
                CausalEvidenceMemory.from_state_dict(state["risk"])
                if instance.evidence_rule
                else CausalPeakMemory.from_state_dict(state["risk"])
            )
            if state["risk"] is not None
            else None
        )
        instance.base_history = deque(state["base_history"], maxlen=instance.packet["capacity"])
        instance.final_history = deque(state["final_history"], maxlen=instance.packet["capacity"])
        instance.last_minute = state["last_minute"]
        return instance
