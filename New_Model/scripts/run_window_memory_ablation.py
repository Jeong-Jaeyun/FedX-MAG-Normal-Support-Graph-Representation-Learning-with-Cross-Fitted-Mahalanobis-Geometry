"""Fixed-energy normal-memory ablation; all raw/deep methods share each budget."""

import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
environment = {**os.environ, "PYTHONPATH": str(root / "New_Model/src")}
experiments = [
    ("round29_cic_context_rank_scan", "round47_cic_rank_buffer200", 200),
    ("round38_xiiotid_reserved_rank_windows", "round48_xiiotid_context_buffer50", 50),
    ("round38_xiiotid_reserved_rank_windows", "round49_xiiotid_context_buffer200", 200),
    ("round45_xiiotid_no_context_rank", "round50_xiiotid_base_buffer50", 50),
    ("round45_xiiotid_no_context_rank", "round51_xiiotid_base_buffer200", 200),
    ("round32_ton_rank_scan", "round52_ton_rank_buffer200", 200),
]
for origin, output, capacity in experiments:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fedxmag_next.window_replay",
            "--run",
            f"New_Model/runs/{origin}",
            "--output",
            f"New_Model/runs/{output}",
            "--buffer",
            str(capacity),
        ],
        cwd=root,
        env=environment,
        check=True,
    )
