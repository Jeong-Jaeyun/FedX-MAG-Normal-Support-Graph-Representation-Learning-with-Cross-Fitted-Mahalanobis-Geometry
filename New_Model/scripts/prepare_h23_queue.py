"""Freeze smoke-verified H2/H3 sources and prepare the sequential full-run queue."""

import json
import os
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
name = "round234_h2_h3_v1"
queue = root / "New_Model/queues" / name
snapshot = root / "New_Model/experiment_sources" / name
scripts = root / "New_Model/scripts"
protocols = (
    "H2_PROTOCOL_2026-11-16.md",
    "H3_PROTOCOL_2026-11-16.md",
)
smoke_audit = root / "New_Model/reports/h23_smoke_audit_2026-11-16_v1/report.json"
verified = json.loads(smoke_audit.read_text(encoding="utf-8"))
if (
    verified.get("status") != "completed"
    or verified.get("smoke") is not True
    or verified.get("total_policy_conditions_checked") != 513
):
    raise ValueError("The six real-data smoke runs must pass independent audit first")
for hypothesis, round_number in (("h2", 234), ("h3", 235)):
    for dataset in ("cic", "xiiotid", "ton"):
        output = root / f"New_Model/runs/round{round_number}_{hypothesis}_{dataset}_v1"
        if output.exists():
            raise FileExistsError(f"Refusing existing full-run output: {output}")
analysis = root / "New_Model/reports/h23_full_2026-11-16_v1"
if queue.exists() or snapshot.exists() or analysis.exists():
    raise FileExistsError("Queue, snapshot or analysis output already exists")
extras = {
    "scripts/background_research_queue.py": scripts / "background_research_queue.py",
    "scripts/audit_h23_results.py": scripts / "audit_h23_results.py",
    **{f"protocols/{p}": root / "New_Model/docs" / p for p in protocols},
    "verification/smoke_audit.json": smoke_audit,
    "verification/smoke_runs.json": root / "New_Model/reports/h23_smoke_inputs_2026-11-16.json",
    "verification/preflight.md": root / "New_Model/reports/H23_PREFLIGHT_2026-11-16.md",
}
environment = frozen_environment(root, name, Path(__file__), extra_files=extras)
prefix = Path(sys.prefix)
overrides = {
    "PYTHONPATH": environment["PYTHONPATH"],
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUNBUFFERED": "1",
    "PATH": os.pathsep.join(
        (str(prefix / "Library/bin"), str(prefix / "DLLs"), os.environ["PATH"])
    ),
    "MKL_THREADING_LAYER": "SEQUENTIAL",
    "OMP_NUM_THREADS": "6",
    "MKL_NUM_THREADS": "6",
    "OPENBLAS_NUM_THREADS": "6",
    "NUMEXPR_NUM_THREADS": "6",
}
queue.mkdir(parents=True, exist_ok=False)
jobs, runs = [], []
for hypothesis, round_number in (("h2", 234), ("h3", 235)):
    for dataset in ("cic", "xiiotid", "ton"):
        output = root / f"New_Model/runs/round{round_number}_{hypothesis}_{dataset}_v1"
        jobs.append(
            {
                "id": f"{hypothesis}_{dataset}",
                "argv": [
                    "-m",
                    f"fedxmag_next.{hypothesis}_experiment",
                    "--root",
                    str(root),
                    "--dataset",
                    dataset,
                    "--output",
                    str(output),
                    "--workers",
                    "6",
                ],
                "expected_report": str(output / "report.json"),
            }
        )
        runs.append({"hypothesis": hypothesis, "dataset": dataset, "run": str(output)})
run_manifest = queue / "runs.json"
with run_manifest.open("x", encoding="utf-8") as handle:
    json.dump(runs, handle, indent=2)
jobs.append(
    {
        "id": "independent_audit",
        "argv": [
            str(snapshot / "scripts/audit_h23_results.py"),
            "--root",
            str(root),
            "--runs",
            str(run_manifest),
            "--output",
            str(analysis),
        ],
        "expected_report": str(analysis / "report.json"),
    }
)
manifest = queue / "manifest.json"
with manifest.open("x", encoding="utf-8") as handle:
    json.dump(
        {"root": str(root), "python": sys.executable, "environment": overrides, "jobs": jobs},
        handle,
        indent=2,
    )
print(
    json.dumps(
        {
            "manifest": str(manifest),
            "runner": str(snapshot / "scripts/background_research_queue.py"),
            "jobs": len(jobs),
            "workers": 6,
        },
        indent=2,
    )
)
