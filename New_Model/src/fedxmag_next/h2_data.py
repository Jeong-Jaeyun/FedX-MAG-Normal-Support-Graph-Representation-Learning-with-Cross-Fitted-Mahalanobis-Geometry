"""Verified H1 source roles and normal-only provenance-group negatives for H2."""

import json
from pathlib import Path

import numpy as np
import torch

from .contrastive_energy import corrupt
from .experiments import sha256
from .h1_classical_features import SourcePCA
from .h1_experiment import PACKETS
from .record_data import NUMERIC
from .record_experiment import corruption_groups


def conditioning_columns(names):
    protocol = [i for i, n in enumerate(names) if n.startswith("protocol_")]
    service = [i for i, n in enumerate(names) if n.startswith("service_")]
    available = [
        i
        for i, n in enumerate(names)
        if n.startswith("available_") or n.endswith("_available") or n == "payload_http_observed"
    ]
    return protocol + service + available, protocol + service


def negative_rows(values, groups, names, policy, seed):
    """All donors are from the supplied normal pool, never from evaluation data.

    Both policies use the whole supplied pool per epoch. Original calls the
    historical permutation algorithm verbatim; conditional adds cohort matching.
    Each donor's complete provenance group is copied, including availability.
    """
    if (
        values.ndim != 2
        or values.shape[1] != len(names)
        or values.device.type != "cpu"
        or not torch.isfinite(values).all()
    ):
        raise ValueError("Corruption requires a finite CPU matrix in original feature coordinates")
    if sorted(i for group in groups for i in group) != list(range(len(names))):
        raise ValueError("Corruption groups must partition the original feature schema")
    generator = torch.Generator().manual_seed(seed)
    if policy == "original":
        return corrupt(values, groups, generator)
    if policy != "conditional":
        raise ValueError("Unknown corruption policy")
    exact_columns, fixed_columns = conditioning_columns(names)
    if not exact_columns or not fixed_columns:
        raise ValueError("Conditional corruption needs explicit schema metadata")
    exact = np.unique(values[:, exact_columns].numpy(), axis=0, return_inverse=True)[1]
    pools = [np.flatnonzero(exact == c) for c in np.unique(exact)]
    changed = values.clone()
    for group in groups:
        # Never relax protocol or service. For an exact-cohort singleton only,
        # relax availability outside the copied group, which is not changed.
        parent_columns = fixed_columns + [
            i for i in group if i in exact_columns and i not in fixed_columns
        ]
        parent = np.unique(values[:, parent_columns].numpy(), axis=0, return_inverse=True)[1]
        eligible = []
        for pool in pools:
            donors = pool if len(pool) > 1 else np.flatnonzero(parent == parent[pool[0]])
            eligible.append((torch.from_numpy(pool), torch.from_numpy(donors)))
        order = torch.arange(len(values))
        for destinations, pool in eligible:
            if len(destinations) == len(pool):
                order[destinations] = pool[torch.randperm(len(pool), generator=generator)]
            else:
                # Still singleton after the parent fallback means a self-donor.
                order[destinations] = pool[
                    torch.randint(len(pool), (len(destinations),), generator=generator)
                ]
        chosen = torch.nonzero(torch.rand(len(values), generator=generator) < 0.5).flatten()
        changed[chosen[:, None], torch.tensor(group)[None, :]] = values[order[chosen]][:, group]
    return changed


def corruption_audit(original, negative, groups, names):
    """Check real encoded invariants, without claiming every network law is known."""
    if not torch.isfinite(negative).all():
        raise ValueError("Nonfinite synthetic observation")
    onehot = {}
    for prefix in ("protocol_", "service_", "state_", "destination_port_"):
        cols = [i for i, n in enumerate(names) if n.startswith(prefix)]
        good = ((negative[:, cols] == 0) | (negative[:, cols] == 1)).all(1)
        good &= negative[:, cols].sum(1) == 1
        if not good.all():
            raise ValueError(f"Broken one-hot invariant: {prefix}")
        onehot[prefix] = True
    numeric = {}
    for name in NUMERIC:
        value, available = names.index("log_" + name), names.index("available_" + name)
        good = (negative[:, value] >= 0) & (
            (negative[:, available] == 1) | (negative[:, value] == 0)
        )
        good &= (negative[:, available] == 0) | (negative[:, available] == 1)
        if not good.all():
            raise ValueError(f"Broken numeric value/availability invariant: {name}")
        numeric[name] = True
    # Every entire group must be a donor row already present in the normal pool.
    for group in groups:
        pool = {tuple(row) for row in original[:, group].tolist()}
        if not all(tuple(row) in pool for row in negative[:, group].tolist()):
            raise ValueError("Synthetic provenance group has no normal donor")
    return {
        "onehot_valid": onehot,
        "numeric_availability_valid": numeric,
        "all_groups_are_normal_donor_groups": True,
        "changed_row_fraction": float((original != negative).any(1).double().mean()),
        "scope": (
            "encoded one-hot, numeric availability and whole-provenance invariants; "
            "not all cross-group physical laws"
        ),
    }


class H2Data:
    def __init__(self, root, dataset, smoke=False):
        root = Path(root).resolve()
        parent = root / f"New_Model/runs/round233_h1_{dataset}_pilot_v2"
        reg = json.loads((parent / "registration.json").read_text())
        report = json.loads((parent / "report.json").read_text())
        if report["status"] != "completed":
            raise ValueError("H1 source contract incomplete")
        packet_path = (
            root / "New_Model/candidates/prototype_structure_v3" / (PACKETS[dataset] + ".pt")
        )
        packet = torch.load(packet_path, weights_only=True)
        risk = root / packet["prototype_model"]["protocol"]["args"]["run"]
        origin = json.loads((risk / "registration.json").read_text())
        self.data_path = root / origin["args"]["data"]
        digest = sha256(self.data_path)
        if digest != reg["data_sha256"] or sha256(packet_path) != reg["packet_sha256"]:
            raise ValueError("H1 data or source contract changed")
        with np.load(self.data_path) as data:
            # Keep the source storage dtype; the normal adapter converts only
            # each requested batch to float64, avoiding a second full-table copy.
            self.features = torch.from_numpy(data["features"])
            self.names = data["feature_names"].tolist()
            self.timestamps, self.row_labels = data["observed_at"], data["label"]
            self.signatures = data["signature"]
        saved = torch.load(parent / "source_adapter.pt", weights_only=True)
        census = torch.load(parent / "scores.pt", weights_only=True)
        self.adapter = SourcePCA.from_state_dict(saved["adapter"])
        self.roles = {k: v.numpy() for k, v in saved["source_role_indices"].items()}
        joined = np.concatenate(list(self.roles.values()))
        self.minutes, self.labels = census["minutes"].numpy(), census["labels"].numpy()
        self.types = census["types"]
        self.cal_minutes = census["source_calibration_minutes"].numpy()
        if (
            len(np.unique(joined)) != len(joined)
            or self.row_labels[joined].any()
            or self.timestamps[joined].max() >= self.minutes.min() * 60
        ):
            raise ValueError("Leaking source roles")
        self.fit = saved["fit_indices"].numpy()
        expected_fit = self.roles["train"][
            np.argsort(self.signatures[self.roles["train"]], kind="stable")[: reg["training_rows"]]
        ]
        if not np.array_equal(self.fit, expected_fit):
            raise ValueError("Changed or non-train H1 fitting indices")
        if len(self.minutes) != len(np.unique(self.minutes)) or not np.all(
            np.diff(self.minutes) > 0
        ):
            raise ValueError("Target chronology is not strictly increasing")
        if (
            not np.all(np.diff(self.cal_minutes) > 0)
            or self.cal_minutes.max() >= self.minutes.min()
        ):
            raise ValueError("Calibration chronology differs or reaches target")
        self.selection = self.roles["selection"]
        self.validation = self.selection[
            np.argsort(self.signatures[self.selection], kind="stable")[:4096]
        ]
        self.combined = saved["combined_indices"].numpy()
        expected = np.concatenate(
            (
                self.roles["calibration"],
                np.flatnonzero(np.isin(self.timestamps // 60, self.minutes)),
            )
        )
        if not np.array_equal(np.sort(expected), np.sort(self.combined)):
            raise ValueError("Changed full prepared target census")
        if len(np.unique(self.combined)) != len(self.combined):
            raise ValueError("Duplicated prepared target/calibration observation")
        if tuple(self.names) != tuple(packet["prototype_model"]["base_encoder"]["feature_names"]):
            raise ValueError("Changed input schema")
        for minute in self.cal_minutes:
            ids = np.flatnonzero(self.timestamps // 60 == minute)
            if self.row_labels[ids].any() or not np.isin(ids, self.roles["calibration"]).all():
                raise ValueError("Partial or contaminated source calibration window")
        self.parent = parent
        self.groups = corruption_groups(self.names)
        self.baselines = census["methods"]
        if smoke:
            # Explicitly labelled integration smoke; never reported as full accuracy.
            keep = np.sort(
                np.r_[np.flatnonzero(self.labels == 0)[:2], np.flatnonzero(self.labels == 1)[:2]]
            )
            self.minutes, self.labels = self.minutes[keep], self.labels[keep]
            self.types = [self.types[i] for i in keep]
            self.fit, self.validation, self.selection = (
                self.fit[:128],
                self.validation[:128],
                self.selection[:128],
            )
            self.cal_minutes = self.cal_minutes[-25:]
            self.combined = np.concatenate(
                [
                    self.combined[self.timestamps[self.combined] // 60 == minute][:64]
                    for minute in np.r_[self.cal_minutes, self.minutes]
                ]
            )
            self.adapter = SourcePCA.fit(self.features[self.fit])
        self.allminutes = np.r_[self.cal_minutes, self.minutes]
        row_minutes = self.timestamps[self.combined] // 60
        order = np.argsort(row_minutes, kind="stable")
        distinct, counts = np.unique(row_minutes[order], return_counts=True)
        self.window_rows = dict(
            zip(distinct.tolist(), np.split(order, np.cumsum(counts)[:-1]), strict=True)
        )
        self.hashes = {
            "data_sha256": digest,
            "h1_registration_sha256": sha256(parent / "registration.json"),
            "source_adapter_sha256": sha256(parent / "source_adapter.pt"),
            "parent_scores_sha256": sha256(parent / "scores.pt"),
        }

    def windows(self, row_scores):
        return np.array(
            [np.sort(row_scores[self.window_rows[int(m)]])[-3:].mean() for m in self.allminutes]
        )
