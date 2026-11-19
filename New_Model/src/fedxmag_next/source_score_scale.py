"""Fixed source-normal score units conditional on observable service equality."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


class SourceScoreScale:
    def __init__(self, state):
        if state.get("format_version") != 1 or state["policy"] not in (
            "global",
            "service",
            "source_service",
        ):
            raise ValueError("Unsupported source-only score scaling state")
        self.state = state
        self.policy, self.global_q95 = state["policy"], state["global_q95"]
        self.service = {tuple(key): value for key, value, count in state["service_entries"]}
        self.source_service = {
            tuple(key): value for key, value, count in state["source_service_entries"]
        }

    @classmethod
    def fit(
        cls, values, source_service_keys, policy, minimum=64, shrinkage=64.0, *, global_q95=None
    ):
        values = np.asarray(values)
        if (
            values.ndim != 1
            or len(values) != len(source_service_keys)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Source-only score samples and observed groups must align")
        if minimum < 2 or shrinkage <= 0:
            raise ValueError("Source-only score profiles need positive normal support budgets")
        global_q95 = float(np.quantile(values, 0.95) if global_q95 is None else global_q95)
        if not np.isfinite(global_q95) or global_q95 <= 0:
            raise ValueError("Source-only score profiles require a positive original q95")
        service_rows, source_rows = defaultdict(list), defaultdict(list)
        for key, value in zip(source_service_keys, values, strict=True):
            service_rows[tuple(key[1:])].append(float(value))
            source_rows[tuple(key)].append(float(value))
        service, source = [], []
        for key, rows in service_rows.items():
            if len(rows) >= minimum:
                q95 = float(np.quantile(rows, 0.95))
                if q95 > 0:
                    value = (len(rows) * q95 + shrinkage * global_q95) / (len(rows) + shrinkage)
                    service.append((key, value, len(rows)))
        service_lookup = {key: value for key, value, count in service}
        for key, rows in source_rows.items():
            if len(rows) >= minimum:
                q95 = float(np.quantile(rows, 0.95))
                if q95 > 0:
                    parent = service_lookup.get(tuple(key[1:]), global_q95)
                    value = (len(rows) * q95 + shrinkage * parent) / (len(rows) + shrinkage)
                    source.append((key, value, len(rows)))
        return cls(
            {
                "format_version": 1,
                "policy": policy,
                "global_q95": global_q95,
                "minimum": minimum,
                "shrinkage": shrinkage,
                "service_entries": service,
                "source_service_entries": source,
            }
        )

    def transform(self, values, source_service_keys):
        values = np.asarray(values)
        if (
            values.ndim != 1
            or len(values) != len(source_service_keys)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Current scores and observable service groups must align")
        if self.policy == "global":
            return values.copy()  # exact identity control, no floating unit conversion
        scales = []
        for key in source_service_keys:
            threshold = self.service.get(tuple(key[1:]), self.global_q95)
            if self.policy == "source_service":
                threshold = self.source_service.get(tuple(key), threshold)
            scales.append(self.global_q95 / threshold)
        return values * np.array(scales)

    def state_dict(self):
        return self.state
