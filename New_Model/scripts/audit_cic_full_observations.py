"""Audit the full CIC query expansion and source-only entity support coverage."""

from pathlib import Path

import numpy as np
import torch

from fedxmag_next.experiments import sha256, write_json


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "New_Model/artifacts/cic_all_legacy_http_v1/records.npz"
    legacy = root / "New_Model/artifacts/cic_http_context_v1/records.npz"
    key_path = root / "New_Model/artifacts/cic_all_entity_keys_v1/keys.npz"
    with np.load(path) as payload:
        data = {k: payload[k] for k in payload.files}
    with np.load(legacy) as payload:
        old = {k: payload[k] for k in payload.files}
    with np.load(key_path) as payload:
        keys = {k: payload[k] for k in payload.files}
    if not np.array_equal(keys["signature"], data["signature"]):
        raise ValueError("Full-data grouping keys are misaligned")
    order = np.argsort(data["signature"])
    positions = order[np.searchsorted(data["signature"][order], old["signature"])]
    exact = {}
    for name in ("signature", "timestamp", "observed_at", "label", "type", "capture", "features"):
        exact[name] = np.array_equal(data[name][positions], old[name])
        if not exact[name]:
            raise ValueError(f"Legacy observations changed: {name}")
    frozen = torch.load(
        root / "New_Model/runs/round133_cic_dependency_flow/scores.pt", weights_only=True
    )
    current = torch.load(
        root / "New_Model/runs/round157_cic_all_refit_flow/scores.pt", weights_only=True
    )
    for name in ("minutes", "labels"):
        if not torch.equal(frozen[name], current[name]):
            raise ValueError(f"Original target census changed: {name}")
    if frozen["types"] != current["types"]:
        raise ValueError("Original target attack types changed")
    visibility = {}
    for name in np.unique(data["type"][data["label"] == 1]):
        expected = current["minutes"].numpy()[np.array([name in t for t in current["types"]])]
        available = np.unique(
            data["observed_at"][(data["label"] == 1) & (data["type"] == name)] // 60
        )
        visibility[str(name)] = {
            "true_attack_windows": len(expected),
            "windows_with_same_type_observation": int(np.isin(expected, available).sum()),
            "attack_rows": int(((data["label"] == 1) & (data["type"] == name)).sum()),
        }
    detector = torch.load(
        root / "New_Model/runs/round156_cic_all_refit_source/source_detector.pt", weights_only=True
    )
    roles = {k: v.numpy() for k, v in detector["source_role_indices"].items()}
    train = roles["train"]
    train = train[np.argsort(data["signature"][train])[:20000]]
    target = np.isin(data["observed_at"] // 60, current["minutes"].numpy())
    profiles = {}
    for scope in ("source", "destination"):
        values = keys[scope]
        ids, counts = np.unique(values[train], return_counts=True)
        selected_ids, selected_counts = np.unique(values[roles["selection"]], return_counts=True)
        count_map = dict(zip(selected_ids.tolist(), selected_counts.tolist(), strict=True))
        supported = ids[
            np.array(
                [
                    c >= 32 and count_map.get(int(k), 0) >= 32
                    for k, c in zip(ids, counts, strict=True)
                ]
            )
        ]
        profiles[scope] = {
            "source_train_entities": len(ids),
            "entities_with_32_train_and_selection": len(supported),
            "target_normal_supported_fraction": float(
                np.isin(values[target & (data["label"] == 0)], supported).mean()
            ),
            "target_attack_supported_fraction": float(
                np.isin(values[target & (data["label"] == 1)], supported).mean()
            ),
            "type_supported_fraction": {
                str(name): float(np.isin(values[target & (data["type"] == name)], supported).mean())
                for name in np.unique(data["type"][target & (data["label"] == 1)])
            },
        }
    report = {
        "status": "descriptive_audit_not_detector_performance",
        "data_sha256": sha256(path),
        "keys_sha256": sha256(key_path),
        "rows": len(data["signature"]),
        "legacy_rows_exact": exact,
        "original_target_census_exact": True,
        "new_http_summaries_inspected": int(data["http_summary_inspected"].sum()),
        "attack_visibility": visibility,
        "entity_profiles": profiles,
    }
    write_json(root / "New_Model/runs/audit_cic_full_observations_v1.json", report)
    import json

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
