"""Transfer the fixed CIC temporal rule to X-IIoTID and TON, without tuning."""

import subprocess
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
environment = frozen_environment(root, "round164_crossdataset_risk_v1", Path(__file__))
for number, dataset, data, keys, source, control in (
    (
        164,
        "xiiotid",
        "xiiotid_context_all_v2",
        "xiiotid_entity_keys_v1",
        "round35_xiiotid_reserved_normal_source",
        "round58_xiiotid_complete_cal_windows",
    ),
    (
        165,
        "ton",
        "ton_protocol_context_v1",
        "ton_role_entity_keys_v1",
        "round120_ton_protocol_context_source",
        "round121_ton_protocol_context_flow",
    ),
):
    command = [
        sys.executable,
        "-m",
        "fedxmag_next.risk_experiment",
        "--data",
        f"New_Model/artifacts/{data}/records.npz",
        "--keys",
        f"New_Model/artifacts/{keys}/keys.npz",
        "--detector",
        f"New_Model/runs/{source}/source_detector.pt",
        "--control",
        f"New_Model/runs/{control}",
        "--output",
        f"New_Model/runs/round{number}_{dataset}_causal_risk",
        "--workers",
        "6",
    ]
    print("RUN", command, flush=True)
    subprocess.run(command, cwd=root, env=environment, check=True)
