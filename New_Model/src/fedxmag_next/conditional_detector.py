"""Read-only inference for frozen current-evidence conditional density artifacts."""

from pathlib import Path

import torch

from .conditional_support import HistoryConditionalSupport
from .daily_detector import DailyDeepEnergyDetector
from .record_experiment import corruption_groups
from .temporal_context import CONTEXT_FEATURES


class ConditionalCurrentDetector:
    def __init__(self, state: dict):
        if state["format_version"] != 1:
            raise ValueError("Unsupported conditional detector format")
        self.state = state
        self.encoder = DailyDeepEnergyDetector(state["current_encoder"])
        self.current_features = self.encoder.feature_names
        corruption_groups(self.current_features)
        if any(name.startswith("context_") for name in self.current_features):
            raise ValueError("Conditional encoder must contain current evidence only")
        if tuple(state["history_features"]) != CONTEXT_FEATURES:
            raise ValueError("Conditional history schema differs")

    @classmethod
    def load(cls, path: str | Path):
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    @torch.no_grad()
    def score(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != len(self.current_features) + len(
            CONTEXT_FEATURES
        ):
            raise ValueError("Conditional query input schema differs")
        if not len(features) or not torch.isfinite(features).all():
            raise ValueError("Conditional query must contain finite observations")
        parts = [self._score(chunk.cpu().float()) for chunk in features.split(8192)]
        return {name: torch.cat([part[name] for part in parts]) for name in parts[0]}

    def _score(self, features):
        state, encoder = self.state, self.encoder.state
        dimension = len(self.current_features)
        current = self.encoder.normalize_features(features[:, :dimension])
        history = (features[:, dimension:] - state["history_mean"]) / state["history_scale"]
        results, deep, marginal_deep = {}, {1: [], 8: []}, {1: [], 8: []}
        for modes in (1, 8):
            name = f"raw_conditional_m{modes}"
            support = HistoryConditionalSupport.from_state_dict(state["conditional_supports"][name])
            results[name] = support.score(current, history)
            results[f"raw_joint_marginal_m{modes}"] = support.marginal_score(current)
        for model, item in zip(self.encoder.models, encoder["seed_models"], strict=True):
            embedded = model.layers[:-1](current)
            embedded = (embedded - item["embedding_mean"]) / item["embedding_scale"]
            for modes in (1, 8):
                fitted = state["conditional_supports"][f"seed_{item['seed']}_m{modes}"]
                support = HistoryConditionalSupport.from_state_dict(fitted["support"])
                energy = support.score(embedded, history)
                deep[modes].append((energy - fitted["center"]) / fitted["scale"] + 1)
                marginal_deep[modes].append(
                    (support.marginal_score(embedded) - fitted["marginal_center"])
                    / fitted["marginal_scale"]
                    + 1
                )
        for modes, values in deep.items():
            results[f"deep_conditional_m{modes}_ensemble"] = torch.stack(values).mean(0)
            results[f"deep_joint_marginal_m{modes}_ensemble"] = torch.stack(
                marginal_deep[modes]
            ).mean(0)
        return results
