"""Attach actual full-API evidence to immutable candidate packets."""

import json
from pathlib import Path

from fedxmag_next.experiments import sha256, write_json

root = Path(__file__).resolve().parents[2]
folder = root / "New_Model/candidates/prototype_structure_v3"
destination = folder / "validated_replays.json"
if destination.exists():
    raise ValueError("Preserve the recorded complete replay validation")
registry = json.loads((folder / "registry.json").read_text())
reports = [
    root / f"New_Model/runs/{name}/report.json"
    for name in (
        "round193_exported_candidate_replay",
        "round213_evidence_candidate_replay",
        "round220_remaining_packet_replay",
    )
]
evidence = {}
for path in reports:
    report = json.loads(path.read_text())
    for candidate in report["candidates"]:
        if (
            candidate["flags_different"]
            or candidate["validity_different"]
            or not candidate["midpoint_disk_restore_actual_inference_and_feedback_exact"]
        ):
            raise ValueError("Recorded actual candidate replay did not pass")
        evidence[candidate["packet_sha256"]] = {
            **candidate,
            "replay_report": str(path),
            "replay_report_sha256": sha256(path),
        }
records = []
for candidate in registry["candidates"]:
    path = Path(candidate["model"])
    if sha256(path) != candidate["model_sha256"]:
        raise ValueError("Registered candidate file changed after its audit")
    proof = evidence.get(candidate["model_sha256"])
    if proof is None:
        raise ValueError(f"No actual initial-state replay for exported packet: {path.name}")
    if any(proof[field] != candidate["metrics"][field] for field in ("tpr", "fpr", "coverage")):
        raise ValueError("Registered candidate metrics differ from its actual API replay")
    records.append(
        {
            "model": str(path),
            "model_sha256": candidate["model_sha256"],
            "actual_initial_state_replay": proof,
            "meets_point_target": proof["tpr"] >= 0.95 and proof["fpr"] <= 0.05,
        }
    )
write_json(
    destination,
    {
        "status": "all_v3_packets_have_actual_full_initial_state_api_replay",
        "goal": "CIC still below95%; TON and X meet point criterion",
        "candidates": records,
    },
)
print(json.dumps({"validated_packets": len(records), "destination": str(destination)}, indent=2))
