"""Observed-service normal geometry and two-sided energy typicality.

Quantiles are fitted on held-out source normals. They are score units, not
conformal probabilities. A separate whole-normal-window buffer sets alarms.
"""

from __future__ import annotations

import torch

from .daily_detector import DailyDeepEnergyDetector, support_energy
from .geometry import ConditionalMahalanobis


def empirical_typicality(energy, sorted_reference):
    """Inclusive tails with1e-6 relative/1e-9 absolute numerical tie tolerance."""
    if len(sorted_reference) == 0:
        raise ValueError("Empirical energy ranks require held-out normal observations")
    tolerance = 1e-6 * energy.abs() + 1e-9
    left = torch.searchsorted(sorted_reference, energy - tolerance, side="left")
    right = torch.searchsorted(sorted_reference, energy + tolerance, side="right")
    upper = (1 + len(sorted_reference) - left).double() / (len(sorted_reference) + 1)
    lower = (1 + right).double() / (len(sorted_reference) + 1)
    two_sided = (2 * torch.minimum(upper, lower)).clamp_max(1)
    return {"upper": -upper.log(), "two_sided": -two_sided.log()}


class CohortEnergyBank:
    def __init__(self, entries):
        self.entries = entries

    @classmethod
    def fit(cls, reference, selection, reference_cohorts, selection_cohorts, modes):
        if len(reference) != len(reference_cohorts) or len(selection) != len(selection_cohorts):
            raise ValueError("Cohort geometry requires aligned normal rows")
        if not torch.isfinite(reference).all() or not torch.isfinite(selection).all():
            raise ValueError("Normal cohort geometry requires finite observations")
        entries = {}
        groups = [-1] + torch.unique(reference_cohorts).tolist()
        for group in groups:
            r = reference if group == -1 else reference[reference_cohorts == group]
            s = selection if group == -1 else selection[selection_cohorts == group]
            if min(len(r), len(s)) < 32:
                continue
            support = ConditionalMahalanobis.fit(r, max_components=modes)
            energy = support_energy(support, s)
            quantiles = torch.quantile(energy, torch.tensor([0.05, 0.5, 0.95]).double())
            entries[str(group)] = {
                "geometry": support.state_dict(),
                "quantiles": quantiles,
                "selection_energy_sorted": energy.sort().values,
                "reference_count": len(r),
                "selection_count": len(s),
            }
        if "-1" not in entries:
            raise ValueError("A global fallback requires at least32 reference/selection normals")
        return cls(entries)

    @torch.no_grad()
    def score(self, values, cohorts, *, conditioned, style="quantile"):
        if style not in ("quantile", "rank"):
            raise ValueError("Unknown normal energy score style")
        if len(values) != len(cohorts):
            raise ValueError("Query cohorts do not match query rows")
        upper = torch.empty(len(values), dtype=torch.float64)
        typicality = torch.empty_like(upper)
        for group in torch.unique(cohorts).tolist():
            mask = cohorts == group
            entry = (
                self.entries.get(str(group), self.entries["-1"])
                if conditioned
                else self.entries["-1"]
            )
            support = ConditionalMahalanobis.from_state_dict(entry["geometry"])
            energy = support_energy(support, values[mask])
            if style == "rank":
                if "selection_energy_sorted" not in entry:
                    raise ValueError("This legacy bank has no saved normal energy ranks")
                ranked = empirical_typicality(energy, entry["selection_energy_sorted"])
                upper[mask], typicality[mask] = ranked["upper"], ranked["two_sided"]
                continue
            low, center, high = entry["quantiles"]
            high_distance = (energy - center) / (high - center).clamp_min(1e-6)
            low_distance = (center - energy) / (center - low).clamp_min(1e-6)
            upper[mask] = high_distance
            typicality[mask] = torch.maximum(high_distance, low_distance)
        return {"upper": upper, "two_sided": typicality}


class CohortEnergyDetector:
    def __init__(self, state):
        if state.get("format_version") != 1:
            raise ValueError("Unsupported cohort energy artifact")
        self.state = state
        self.base = DailyDeepEnergyDetector(state["base_encoder"])

    @classmethod
    def load(cls, path):
        return cls(torch.load(path, map_location="cpu", weights_only=True))

    @torch.no_grad()
    def score(self, features):
        from .context_association import cohort_ids

        if features.ndim != 2 or features.shape[1] != len(self.base.feature_names):
            raise ValueError("Cohort query schema differs from its frozen encoder")
        if not torch.isfinite(features).all():
            raise ValueError("Cohort queries must be finite")
        cohorts = cohort_ids(features)
        values = self.base.normalize_features(features)
        encodings = {"raw": values}
        for model, item in zip(self.base.models, self.base.state["seed_models"], strict=True):
            embedded = (
                torch.cat([model.layers[:-1](chunk) for chunk in values.split(8192)])
                if len(values)
                else item["embedding_mean"].new_empty((0, len(item["embedding_mean"])))
            )
            embedded = (embedded - item["embedding_mean"]) / item["embedding_scale"]
            encodings[f"seed_{item['seed']}"] = embedded
        scored, deep = {}, {}
        style = self.state.get("score_style", "quantile")
        for name, encoding in encodings.items():
            for modes in (1, 8):
                bank = CohortEnergyBank(self.state["banks"][f"{name}_m{modes}"])
                for scope in ("global", "cohort"):
                    for tail, energy in bank.score(
                        encoding, cohorts, conditioned=scope == "cohort", style=style
                    ).items():
                        prefix = f"{scope}_rank" if style == "rank" else scope
                        key = f"{prefix}_{tail}_m{modes}"
                        if name == "raw":
                            scored[f"raw_{key}"] = energy
                        else:
                            deep.setdefault(key, []).append(energy)
        for key, energies in deep.items():
            scored[f"deep_{key}_ensemble"] = torch.stack(energies).mean(0)
        return scored
