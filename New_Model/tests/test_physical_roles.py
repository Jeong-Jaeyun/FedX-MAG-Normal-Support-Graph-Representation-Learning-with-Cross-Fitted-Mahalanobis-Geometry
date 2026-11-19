import json

import numpy as np
import pytest
import torch

from fedxmag_next.experiments import sha256
from fedxmag_next.record_data import FEATURES
from fedxmag_next.source_only import physical_role_template


def test_new_observations_keep_physical_roles_and_long_flows_still_respect_purge(tmp_path):
    times = np.r_[1000 + np.arange(40) * 10, 2000 + np.arange(40) * 10, 3000 + np.arange(40) * 10]
    old = {
        "signature": np.arange(120, dtype=np.uint64),
        "timestamp": times.copy(),
        "observed_at": times.copy(),
        "dataset": np.array("ton"),
        "feature_names": np.array(FEATURES),
        "label": np.zeros(120, dtype=np.int8),
    }
    data_path = tmp_path / "old.npz"
    np.savez(data_path, **old)
    roles = {
        name: torch.arange(i * 40, (i + 1) * 40)
        for i, name in enumerate(("train", "selection", "calibration"))
    }
    artifact = tmp_path / "source.pt"
    torch.save({"source_data_sha256": sha256(data_path), "source_role_indices": roles}, artifact)
    (tmp_path / "registration.json").write_text(
        json.dumps({"args": {"data": str(data_path), "history_gap": 300}})
    )
    new = {
        **old,
        "signature": np.r_[old["signature"], np.arange(1000, 1006, dtype=np.uint64)],
        "timestamp": np.r_[times, 1055, 2055, 3055, 1750, 1300, 1056],
        "observed_at": np.r_[times, 1055, 2055, 3055, 1750, 2300, 1056],
        "label": np.r_[np.zeros(125, dtype=np.int8), 1],
    }
    assigned, policy = physical_role_template(new, str(artifact), 300)
    assert policy["all_original_role_observations_retained"]
    for i, name in enumerate(("train", "selection", "calibration")):
        assert len(assigned[name]) == 41
        assert 120 + i in assigned[name]
    assert all(not np.isin([123, 124, 125], rows).any() for rows in assigned.values())
    extended = {**old, "feature_names": np.concatenate((old["feature_names"], ["payload_extra"]))}
    with pytest.raises(ValueError, match="original feature schema"):
        physical_role_template(extended, str(artifact), 300)
    matched, _ = physical_role_template(extended, str(artifact), 300, allow_feature_change=True)
    assert all(np.array_equal(matched[name], rows.numpy()) for name, rows in roles.items())
    with pytest.raises(ValueError, match="keep every observation"):
        physical_role_template(
            new | {"feature_names": extended["feature_names"]},
            str(artifact),
            300,
            allow_feature_change=True,
        )


def test_same_observations_preserve_row_membership_when_a_boundary_second_is_tied(tmp_path):
    times = np.r_[
        1000 + np.arange(40) * 10, 1390, 2000 + np.arange(40) * 10, 2390, 3000 + np.arange(40) * 10
    ]
    data = {
        "signature": np.arange(len(times), dtype=np.uint64),
        "timestamp": times.copy(),
        "observed_at": times.copy(),
        "dataset": np.array("ton"),
        "feature_names": np.array(FEATURES),
        "label": np.zeros(len(times), dtype=np.int8),
    }
    roles = {
        "train": torch.arange(40),
        "selection": torch.arange(41, 81),
        "calibration": torch.arange(82, 122),
    }
    data_path = tmp_path / "data.npz"
    np.savez(data_path, **data)
    artifact = tmp_path / "source.pt"
    torch.save({"source_data_sha256": sha256(data_path), "source_role_indices": roles}, artifact)
    (tmp_path / "registration.json").write_text(
        json.dumps({"args": {"data": str(data_path), "history_gap": 300}})
    )
    for extended in (False, True):
        query = data.copy()
        if extended:
            query["feature_names"] = np.concatenate((data["feature_names"], ["payload_extra"]))
        assigned, policy = physical_role_template(query, str(artifact), 300, extended)
        assert policy["membership_policy"] == "exact_saved_indices"
        assert all(np.array_equal(assigned[name], rows.numpy()) for name, rows in roles.items())
        assert all(not np.isin([40, 81], rows).any() for rows in assigned.values())
