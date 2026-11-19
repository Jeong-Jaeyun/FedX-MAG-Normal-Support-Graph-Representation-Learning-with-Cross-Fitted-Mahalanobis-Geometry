"""Read-only cross-modal clock/coverage audit. No model or joined feature fit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fedxmag_next.experiments import sha256

root = Path(__file__).resolve().parents[2]
folder = root / "data/TON-IoT/Processed_datasets"
control = root / "New_Model/runs/round121_ton_protocol_context_flow/scores.pt"
target = torch.load(control, weights_only=True)
minutes, labels = target["minutes"].numpy(), target["labels"].numpy()
types = target["types"]
paths = sorted(
    path
    for modality in (
        "Processed_IoT_dataset",
        "Processed_Linux_dataset",
        "Processed_Windows_dataset",
    )
    for path in (folder / modality).glob("*.csv")
)
report = {
    "status": "retrospective_clock_coverage_audit_only_no_model",
    "control_sha256": sha256(control),
    "clock_contract": (
        "epoch ts used literally; IoT date/time provisionally parsed as UTC, "
        "consistent with sampled raw JSON GMT strings, not a calibrated cross-clock offset"
    ),
    "endpoint_mapping": "no validated telemetry-id to network-endpoint mapping",
    "model_inputs_changed": False,
    "files": {},
}
unions = {}
for path in paths:
    header = pd.read_csv(path, nrows=0).columns
    fields = ["ts"] if "ts" in header else ["date", "time"]
    observed_minutes, attack_ranges = set(), {}
    total, invalid, minimum, maximum = 0, 0, None, None
    for frame in pd.read_csv(
        path,
        usecols=fields + ["label", "type"],
        dtype=str,
        keep_default_na=False,
        chunksize=250000,
    ):
        total += len(frame)
        if fields == ["ts"]:
            timestamp = pd.to_numeric(frame["ts"], errors="coerce").to_numpy()
        else:
            parsed = pd.to_datetime(
                frame["date"].str.strip() + " " + frame["time"].str.strip(),
                format="%d-%b-%y %H:%M:%S",
                errors="coerce",
                utc=True,
            )
            # pandas3 selects microsecond storage for these strings. Normalize
            # the unit explicitly rather than assuming the integer is ns.
            timestamp = parsed.dt.as_unit("s").astype("int64").to_numpy(dtype=float)
            timestamp[parsed.isna().to_numpy()] = np.nan
        valid = np.isfinite(timestamp)
        invalid += int((~valid).sum())
        timestamp = timestamp[valid].astype(np.int64)
        frame = frame.loc[valid]
        if not len(timestamp):
            continue
        observed_minutes.update((timestamp // 60).tolist())
        minimum = int(timestamp.min()) if minimum is None else min(minimum, int(timestamp.min()))
        maximum = int(timestamp.max()) if maximum is None else max(maximum, int(timestamp.max()))
        for kind in np.unique(frame["type"]):
            selected = timestamp[frame["type"].to_numpy() == kind]
            entry = attack_ranges.setdefault(
                kind, {"rows": 0, "min_ts": int(selected.min()), "max_ts": int(selected.max())}
            )
            entry["rows"] += len(selected)
            entry["min_ts"] = min(entry["min_ts"], int(selected.min()))
            entry["max_ts"] = max(entry["max_ts"], int(selected.max()))
    observed = np.isin(minutes, list(observed_minutes))
    modality = path.parent.name
    unions.setdefault(modality, np.zeros(len(minutes), dtype=bool))
    unions[modality] |= observed
    entry = {
        "sha256": sha256(path),
        "rows": total,
        "invalid_times": invalid,
        "min_ts": minimum,
        "max_ts": maximum,
        "per_record_type": attack_ranges,
        "same_minute_target_coverage": {
            "normal": float(observed[labels == 0].mean()),
            "attack": float(observed[labels == 1].mean()),
            "mitm": float(observed[np.array(["mitm" in group for group in types])].mean()),
        },
    }
    report["files"][path.name] = entry
    print(path.name, total, entry["same_minute_target_coverage"], flush=True)
report["union_same_minute_coverage"] = {
    key: {"normal": float(value[labels == 0].mean()), "attack": float(value[labels == 1].mean())}
    for key, value in unions.items()
}
output = root / "New_Model/runs/audit_ton_telemetry_alignment_v2.json"
with output.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
