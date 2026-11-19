"""Prepare synthetic diagnostics behind the existing H2/H3 CPU-worker gate."""

import hashlib
import json
import os
import sys
from pathlib import Path

from experiment_snapshot import frozen_environment

root = Path(__file__).resolve().parents[2]
name = "round236_h23_diagnostics_v1"
scripts = root / "New_Model/scripts"
queue = root / "New_Model/queues" / name
snapshot = root / "New_Model/experiment_sources" / name
upstream = root / "New_Model/queues/round234_h2_h3_v1/manifest.json"
smoke = root / "New_Model/runs/round236_h23_mechanisms_smoke_v2"
smoke_audit = root / "New_Model/reports/h23_mechanisms_smoke_audit_2026-11-16_v1/report.json"
verified = json.loads(smoke_audit.read_text(encoding="utf-8"))
if (
    verified.get("status") != "completed"
    or verified.get("smoke") is not True
    or verified.get("total_policy_conditions_checked") != 1152
):
    raise ValueError("All synthetic diagnostic smoke conditions must pass independent audit")
registered = json.loads((smoke / "registration.json").read_text(encoding="utf-8"))
module = root / "New_Model/src/fedxmag_next/h23_mechanisms.py"
if registered["source_sha256"] != hashlib.sha256(module.read_bytes()).hexdigest():
    raise ValueError("Diagnostic source changed since the smoke test")
if not upstream.is_file():
    raise FileNotFoundError("The upstream queue must already exist")
full = root / "New_Model/runs/round236_h23_mechanisms_v1"
findings = root / "New_Model/reports/h23_findings_2026-11-16_v1"
analysis = root / "New_Model/reports/h23_mechanisms_full_2026-11-16_v1"
if any(p.exists() for p in (queue, snapshot, full, findings, analysis)):
    raise FileExistsError("Refusing to overwrite a queue, source snapshot or result")
extra = {
    **{
        f"scripts/{p}": scripts / p
        for p in (
            "background_research_queue.py",
            "wait_research_queue.py",
            "audit_h23_results.py",
            "audit_h23_mechanisms.py",
            "summarize_h23_findings.py",
        )
    },
    "protocol.md": root / "New_Model/docs/H23_MECHANISMS_PROTOCOL_2026-11-16.md",
    "verification/smoke_audit.json": smoke_audit,
    "verification/smoke_registration.json": smoke / "registration.json",
}
env = frozen_environment(root, name, Path(__file__), extra_files=extra)
queue.mkdir(parents=True, exist_ok=False)
prefix = Path(sys.prefix)
overrides = {
    "PYTHONPATH": env["PYTHONPATH"],
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
gate = queue / "upstream_gate"
jobs = [
    {
        "id": "wait_workers",
        "argv": [
            str(snapshot / "scripts/wait_research_queue.py"),
            "--upstream",
            str(upstream),
            "--output",
            str(gate),
        ],
        "expected_report": str(gate / "report.json"),
    },
    {
        "id": "summarize_real",
        "requires": ["wait_workers"],
        "argv": [
            str(snapshot / "scripts/summarize_h23_findings.py"),
            "--audit",
            str(root / "New_Model/reports/h23_full_2026-11-16_v1/report.json"),
            "--output",
            str(findings),
        ],
        "expected_report": str(findings / "findings.json"),
    },
    {
        "id": "mechanisms",
        "requires": ["wait_workers"],
        "argv": ["-m", "fedxmag_next.h23_mechanisms", "--output", str(full), "--workers", "6"],
        "expected_report": str(full / "report.json"),
    },
    {
        "id": "mechanism_audit",
        "requires": ["mechanisms"],
        "argv": [
            str(snapshot / "scripts/audit_h23_mechanisms.py"),
            "--run",
            str(full),
            "--output",
            str(analysis),
        ],
        "expected_report": str(analysis / "report.json"),
    },
]
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
            "jobs": len(jobs),
            "runner": str(snapshot / "scripts/background_research_queue.py"),
        },
        indent=2,
    )
)
