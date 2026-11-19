"""Run the registered tail extension and matched X context-removal ablation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "New_Model/src"),
        "PYTHONIOENCODING": "utf-8",
    }

    def invoke(module: str, *arguments: str) -> None:
        subprocess.run(
            [sys.executable, "-m", f"fedxmag_next.{module}", *arguments],
            cwd=root,
            env=environment,
            check=True,
        )

    def window(data: str, detector_run: str, output: str, aggregation: str) -> None:
        invoke(
            "window_bridge",
            "--data",
            f"New_Model/artifacts/{data}/records.npz",
            "--detector",
            f"New_Model/runs/{detector_run}/source_detector.pt",
            "--output",
            f"New_Model/runs/{output}",
            "--workers",
            "6",
            "--history-policy",
            "continuous",
            "--aggregation",
            aggregation,
        )

    window(
        "cic_http_context_v1",
        "round25_cic_context_full_prequential",
        "round42_cic_context_tail_scan",
        "tail_scan",
    )
    window(
        "xiiotid_context_all_v2",
        "round35_xiiotid_reserved_normal_source",
        "round43_xiiotid_context_tail_scan",
        "tail_scan",
    )
    window(
        "ton_connections_v2", "round17_ton_full_prequential", "round44_ton_tail_scan", "tail_scan"
    )
    # Same full X rows, source roles, reference cap, seeds and evaluation;
    # remove only the 36 anonymous historical columns in this matched ablation.
    invoke(
        "source_only",
        "--data",
        "New_Model/artifacts/xiiotid_connections_all_v1/records.npz",
        "--output",
        "New_Model/runs/round40_xiiotid_no_context_source",
        "--workers",
        "6",
        "--epochs",
        "30",
        "--role-order",
        "auto_normal_windows",
    )
    window(
        "xiiotid_connections_all_v1",
        "round40_xiiotid_no_context_source",
        "round45_xiiotid_no_context_rank",
        "rank_scan",
    )
    window(
        "xiiotid_connections_all_v1",
        "round40_xiiotid_no_context_source",
        "round46_xiiotid_no_context_tail",
        "tail_scan",
    )


if __name__ == "__main__":
    main()
