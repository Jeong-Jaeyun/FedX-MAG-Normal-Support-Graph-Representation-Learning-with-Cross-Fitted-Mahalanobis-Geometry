"""Independently verify all-array identity after the completed HTTP acquisition."""

import json
from pathlib import Path

import numpy as np
import torch

from fedxmag_next.experiments import sha256, write_json
from fedxmag_next.full_http_observations import URI_COLUMNS
from fedxmag_next.pcap_http import HTTP_FEATURES

root = Path(__file__).resolve().parents[2]
old_path = root / "New_Model/artifacts/cic_all_legacy_http_v1/records.npz"
new_path = root / "New_Model/artifacts/cic_all_packet_http_v1/records.npz"
source_path = root / "New_Model/runs/round186_cic_packet_http_source/source_detector.pt"
output = new_path.parent / "independent_array_audit.json"
if output.exists():
    raise ValueError("Preserve the existing complete-input audit")
with np.load(old_path) as payload:
    old = {key: payload[key] for key in payload.files}
with np.load(new_path) as payload:
    new = {key: payload[key] for key in payload.files}
if old.keys() != new.keys():
    raise ValueError("HTTP acquisition changed the prepared array schema")
changed_arrays = {"features", "http_features", "http_summary_inspected"}
for name in old.keys() - changed_arrays:
    if not np.array_equal(old[name], new[name]):
        raise ValueError(f"HTTP acquisition changed metadata: {name}")
inspected = old["http_summary_inspected"]
if not np.array_equal(old["features"][inspected], new["features"][inspected]):
    raise ValueError("HTTP acquisition changed a previously inspected feature vector")
allowed = set(URI_COLUMNS) | {f"payload_{name}" for name in HTTP_FEATURES}
changes = {}
for i, name in enumerate(old["feature_names"]):
    count = int((old["features"][:, i] != new["features"][:, i]).sum())
    if count:
        if name not in allowed:
            raise ValueError(f"HTTP acquisition changed a non-HTTP column: {name}")
        changes[str(name)] = count
if not new["http_summary_inspected"].all():
    raise ValueError("Some prepared connections remain uninspected")
source = torch.load(source_path, weights_only=True)
previous_source = torch.load(
    root / "New_Model/runs/round156_cic_all_refit_source/source_detector.pt", weights_only=True
)
for name, indices in previous_source["source_role_indices"].items():
    if not torch.equal(indices, source["source_role_indices"][name]):
        raise ValueError(f"HTTP refit changed the saved source role: {name}")
report = {
    "status": "all_array_metadata_and_saved_source_roles_verified_exact",
    "old_sha256": sha256(old_path),
    "new_sha256": sha256(new_path),
    "source_detector_sha256": sha256(source_path),
    "rows": len(inspected),
    "old_inspected_feature_vectors_bit_exact": int(inspected.sum()),
    "newly_inspected": int((~inspected).sum()),
    "all_non_http_columns_exact": True,
    "all_other_metadata_arrays_exact": True,
    "source_role_indices_exact": True,
    "changed_column_row_counts": changes,
    "old_observed_http": int((old["http_features"][:, 0] > 0).sum()),
    "new_observed_http": int((new["http_features"][:, 0] > 0).sum()),
}
write_json(output, report)
print(json.dumps(report, indent=2))
