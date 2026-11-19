"""Frozen deep features with normal support conditioned on endpoint equality.

Endpoint hashes route geometry only; their magnitudes never enter the encoder.
This is explicitly a same-network personalized model, not identity-free transfer.
"""

from __future__ import annotations

import numpy as np
import torch

from .cohort_energy import CohortEnergyBank
from .daily_detector import DailyDeepEnergyDetector


def route_entities(keys, vocabulary):
    """Map uint64 equality keys to compact IDs; unseen keys have no local bank."""
    keys = np.asarray(keys)
    if keys.ndim != 1 or keys.dtype != np.uint64:
        raise ValueError("Entity keys must be one-dimensional uint64 equality metadata")
    vocabulary = vocabulary.numpy().view(np.uint64)
    positions = np.searchsorted(vocabulary, keys)
    found = positions < len(vocabulary)
    found[found] &= vocabulary[positions[found]] == keys[found]
    return torch.from_numpy(np.where(found, positions, len(vocabulary)).astype(np.int64))


class EntitySupportDetector:
    def __init__(self, state):
        if state.get("format_version") != 1:
            raise ValueError("Unsupported entity support artifact")
        self.state = state
        self.base = DailyDeepEnergyDetector(state["base_encoder"])
        if self.base.encoder_kind != "dependency":
            raise ValueError("Entity support requires a dependency encoder")
        for model in self.base.models:
            model.double()

    @classmethod
    def load(cls, path):
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    @torch.no_grad()
    def encodings(self, features):
        if features.ndim != 2 or features.shape[1] != len(self.base.feature_names):
            raise ValueError("Entity support schema differs from frozen encoder")
        if not torch.isfinite(features).all():
            raise ValueError("Entity support needs finite features")
        # Preserve original float32 input units; evaluate copied coefficients in
        # double precision to reduce query-batch-dependent rank perturbations.
        values = self.base.normalize_features(features).double()
        encoded = {"raw": values}
        for model, item in zip(self.base.models, self.base.state["seed_models"], strict=True):
            hidden = (
                torch.cat([model.layers[:-1](chunk) for chunk in values.split(8192)])
                if len(values)
                else values.new_empty((0, len(item["embedding_mean"])))
            )
            encoded[f"seed_{item['seed']}"] = (hidden - item["embedding_mean"].double()) / item[
                "embedding_scale"
            ].double()
        return encoded

    @torch.no_grad()
    def score(self, features, keys):
        encodings = self.encodings(features)
        if any(len(keys[scope]) != len(features) for scope in ("source", "destination")):
            raise ValueError("Entity routing keys do not match query rows")
        routed = {
            scope: route_entities(keys[scope], self.state["vocabulary"][scope])
            for scope in ("source", "destination")
        }
        output, neural = {}, {}
        for name, values in encodings.items():
            for modes in (1, 8):
                for scope in ("global", "source", "destination"):
                    owner = "source" if scope == "global" else scope
                    bank = CohortEnergyBank(self.state["banks"][f"{name}_{owner}_m{modes}"])
                    # Collapse unsupported identities before scoring. Query
                    # cost then depends on fitted profiles, not remote hosts.
                    supported = torch.tensor(
                        [int(k) for k in bank.entries if k != "-1"], dtype=torch.long
                    )
                    ids = (
                        torch.where(torch.isin(routed[owner], supported), routed[owner], -1)
                        if scope != "global"
                        else torch.full_like(routed[owner], -1)
                    )
                    score = bank.score(values, ids, conditioned=scope != "global")["upper"]
                    key = f"entity_{scope}_m{modes}"
                    if name == "raw":
                        output[f"raw_{key}"] = score
                    else:
                        neural.setdefault(key, []).append(score)
        output.update({f"deep_{key}_ensemble": torch.stack(v).mean(0) for key, v in neural.items()})
        return output


@torch.no_grad()
def fit_entity_state(base, reference, selection, reference_keys, selection_keys):
    state = {"format_version": 1, "base_encoder": base, "banks": {}, "vocabulary": {}}
    detector = EntitySupportDetector(state)
    r, s = detector.encodings(reference), detector.encodings(selection)
    for scope in ("source", "destination"):
        vocabulary = np.unique(reference_keys[scope])
        state["vocabulary"][scope] = torch.from_numpy(vocabulary.view(np.int64).copy())
        ri = route_entities(reference_keys[scope], state["vocabulary"][scope])
        si = route_entities(selection_keys[scope], state["vocabulary"][scope])
        for name in r:
            for modes in (1, 8):
                state["banks"][f"{name}_{scope}_m{modes}"] = CohortEnergyBank.fit(
                    r[name], s[name], ri, si, modes
                ).entries
    return state
