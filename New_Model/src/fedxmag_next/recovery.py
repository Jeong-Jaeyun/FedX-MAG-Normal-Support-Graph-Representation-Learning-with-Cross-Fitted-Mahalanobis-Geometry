"""Recover the zero-byte v3b attack artifact from preserved v3 measurements.

No archived file is overwritten. All feature names and scalers are matched by
name; overlapping normal roles are checked before deriving the missing role.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .experiments import sha256, write_json


def recover(root: Path) -> Path:
    old = root / "data/processed/xiiotid_stage1_v3_development"
    new = root / "data/processed/xiiotid_stage1_v3b_development"
    target = root / "New_Model/artifacts/recovered_v3b/test_attack.pt"
    if target.exists():
        return target
    template = torch.load(new / "calibration_normal.pt", weights_only=True)
    old_scaler = torch.load(old / "standardizer.pt", weights_only=True)
    new_scaler = torch.load(new / "standardizer.pt", weights_only=True)
    source = torch.load(old / "test_attack.pt", weights_only=True)
    positions = {
        kind: [source[f"{kind}_features"].index(name) for name in template[f"{kind}_features"]]
        for kind in ("node", "edge")
    }

    def transform(record: dict) -> dict:
        result = dict(record)
        metadata = json.loads(result["metadata_json"])
        metadata["timing_feature_profile"] = "distinct_second_collision_v3"
        result["metadata_json"] = json.dumps(metadata, sort_keys=True)
        for kind in ("node", "edge"):
            index = positions[kind]
            available = record[f"{kind}_available"][:, index]
            raw = (
                record[f"{kind}_features"][:, index] * old_scaler[f"{kind}_scale"][index]
                + old_scaler[f"{kind}_mean"][index]
            )
            result[f"{kind}_features"] = (
                (raw - new_scaler[f"{kind}_mean"]) / new_scaler[f"{kind}_scale"]
            ).masked_fill(~available, 0.0)
            result[f"{kind}_available"] = available
        return result

    checks = {}
    for role in ("calibration_normal", "test_normal", "audit_normal"):
        left = torch.load(old / f"{role}.pt", weights_only=True)["records"]
        right = torch.load(new / f"{role}.pt", weights_only=True)["records"]
        if len(left) != len(right):
            raise ValueError(f"Window count differs for recovery overlap {role}")
        maximum = 0.0
        for earlier, current in zip(left, right, strict=True):
            derived = transform(earlier)
            old_meta, new_meta = (
                json.loads(earlier["metadata_json"]),
                json.loads(current["metadata_json"]),
            )
            for name in ("session_key", "window_start_ns", "endpoint_keys", "edge_keys"):
                if old_meta[name] != new_meta[name]:
                    raise ValueError(f"Recovery alignment differs for {role}/{name}")
            if not torch.equal(earlier["edge_index"], current["edge_index"]):
                raise ValueError("Recovery graph topology differs")
            for kind in ("node", "edge"):
                if not torch.equal(derived[f"{kind}_available"], current[f"{kind}_available"]):
                    raise ValueError("Recovery availability differs")
                maximum = max(
                    maximum,
                    float((derived[f"{kind}_features"] - current[f"{kind}_features"]).abs().max()),
                )
        if maximum > 2e-6:
            raise ValueError(f"Recovery numeric mismatch: {maximum}")
        checks[role] = {"windows": len(left), "max_absolute_feature_difference": maximum}
    payload = {key: value for key, value in template.items() if key != "records"}
    payload["records"] = [transform(record) for record in source["records"]]
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    write_json(
        target.parent / "recovery_report.json",
        {
            "reason": "Original v3b test_attack.pt is zero bytes",
            "source": str(old / "test_attack.pt"),
            "source_sha256": sha256(old / "test_attack.pt"),
            "derived_sha256": sha256(target),
            "checks": checks,
            "attack_windows": len(payload["records"]),
            "note": "Derived measurement-equivalent artifact; not byte-identical to lost archive",
        },
    )
    return target


if __name__ == "__main__":
    print(recover(Path.cwd()), flush=True)
