"""Run an immutable, sequential research queue without a visible child console.

Usage: python background_research_queue.py --queue C:/absolute/path/manifest.json

Manifest schema (all paths absolute)::

    {"root": "D:/FedX_MAG", "python": "C:/env/python.exe",
     "environment": {"PYTHONPATH": "D:/frozen/src", "OMP_NUM_THREADS": "6"},
     "jobs": [{"id": "h2_cic", "argv": ["-m", "package.module", "--output", "..."],
               "expected_report": "D:/output/report.json", "requires": []}]}

Each argv excludes the Python executable. No shell is used. The manifest folder
receives queue.lock, status.json, and logs/<id>.log. Existing queue state is never
resumed or overwritten. A failed job does not prevent independent later jobs.
Optional ``requires`` lists unique earlier job IDs. A dependent job is skipped
unless every requirement has status=completed (zero exit alone is insufficient).
Exit 0 means every job exited 0 and its report says status=completed; exit 1 means
one or more jobs failed; exit 2 means queue setup or infrastructure failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path


def utc_now():
    return datetime.now(UTC).isoformat()


def atomic_status(path, state):
    state["heartbeat_at"] = utc_now()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".status-",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_manifest(path):
    if not path.is_absolute():
        raise ValueError("--queue must be an absolute manifest path")
    raw = path.read_bytes()
    manifest = json.loads(raw)
    for field in ("root", "python"):
        if not isinstance(manifest.get(field), str) or not Path(manifest[field]).is_absolute():
            raise ValueError(f"Manifest {field} must be absolute")
    if not Path(manifest["root"]).is_dir() or not Path(manifest["python"]).is_file():
        raise ValueError("Workspace or Python interpreter does not exist")
    environment = manifest.get("environment", {})
    if not isinstance(environment, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in environment.items()
    ):
        raise ValueError("environment must contain string overrides")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Queue must contain at least one job")
    identifiers, reports = set(), set()
    for job in jobs:
        identifier = job.get("id", "")
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", identifier):
            raise ValueError("Job id must contain only letters, numbers, underscore, or hyphen")
        if identifier in identifiers:
            raise ValueError("Duplicate job id")
        requirements = job.get("requires", [])
        if not isinstance(requirements, list) or not all(
            isinstance(requirement, str) and requirement for requirement in requirements
        ):
            raise ValueError("Job requires must be a list of nonempty earlier job IDs")
        if len(set(requirements)) != len(requirements):
            raise ValueError("Job requires contains duplicate dependency IDs")
        if not set(requirements).issubset(identifiers):
            raise ValueError("Job requires must reference only earlier job IDs")
        identifiers.add(identifier)
        argv = job.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError("Job argv must be a nonempty list of strings")
        report = Path(job["expected_report"])
        if not report.is_absolute() or str(report.resolve()).casefold() in reports:
            raise ValueError("Reports must be distinct absolute paths")
        reports.add(str(report.resolve()).casefold())
    return manifest, hashlib.sha256(raw).hexdigest()


def completed_report(path):
    if not path.is_file():
        return False, "expected_report_missing"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return False, f"expected_report_unreadable: {error}"
    if not isinstance(result, dict) or result.get("status") != "completed":
        return False, "expected_report_not_completed"
    return True, None


def run_queue(manifest_path, heartbeat_seconds=10.0):
    if heartbeat_seconds <= 0:
        raise ValueError("Heartbeat interval must be positive")
    manifest_path = Path(manifest_path)
    manifest, digest = load_manifest(manifest_path)
    folder = manifest_path.parent
    status_path, lock_path = folder / "status.json", folder / "queue.lock"
    if status_path.exists() or lock_path.exists():
        raise FileExistsError("Queue state exists; refusing resume or overwrite")
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "created_at": utc_now(), "manifest_sha256": digest}, handle)
    state = {
        "format_version": 1,
        "status": "running",
        "pid": os.getpid(),
        "started_at": utc_now(),
        "finished_at": None,
        "manifest": str(manifest_path),
        "manifest_sha256": digest,
        "current_job": None,
        "job_pid": None,
        "completed_jobs": [],
        "failed_jobs": [],
        "skipped_jobs": [],
        "jobs": [],
    }
    process = None
    try:
        log_folder = folder / "logs"
        log_folder.mkdir(exist_ok=False)
        for job in manifest["jobs"]:
            state["jobs"].append(
                {
                    "id": job["id"],
                    "status": "queued",
                    "pid": None,
                    "exit_code": None,
                    "started_at": None,
                    "finished_at": None,
                    "error": None,
                    "log": str(log_folder / (job["id"] + ".log")),
                    "expected_report": job["expected_report"],
                    "requires": job.get("requires", []),
                }
            )
        atomic_status(status_path, state)
        environment = {**os.environ, **manifest.get("environment", {})}
        for job, record in zip(manifest["jobs"], state["jobs"], strict=True):
            statuses = {entry["id"]: entry["status"] for entry in state["jobs"]}
            unmet = [name for name in job.get("requires", []) if statuses[name] != "completed"]
            if unmet:
                record.update(
                    status="skipped",
                    finished_at=utc_now(),
                    error="dependency_not_completed: " + ", ".join(unmet),
                )
                state["failed_jobs"].append(job["id"])
                state["skipped_jobs"].append(job["id"])
                with Path(record["log"]).open("x", encoding="utf-8") as log:
                    log.write(record["error"] + "\n")
                atomic_status(status_path, state)
                continue
            state["current_job"] = job["id"]
            record.update(status="running", started_at=utc_now())
            atomic_status(status_path, state)
            with Path(record["log"]).open("xb") as log:
                try:
                    report_path = Path(job["expected_report"])
                    if report_path.exists():
                        raise FileExistsError("Expected report already exists; refusing rerun")
                    command = [manifest["python"], *job["argv"]]
                    process = subprocess.Popen(
                        command,
                        cwd=manifest["root"],
                        env=environment,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        shell=False,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                    record["pid"] = state["job_pid"] = process.pid
                    atomic_status(status_path, state)
                    while True:
                        try:
                            record["exit_code"] = process.wait(timeout=heartbeat_seconds)
                            break
                        except subprocess.TimeoutExpired:
                            atomic_status(status_path, state)
                    process = None
                    if record["exit_code"] != 0:
                        record["error"] = "process_nonzero_exit"
                    else:
                        good, error = completed_report(report_path)
                        record["error"] = None if good else error
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                    log.write(traceback.format_exc().encode("utf-8"))
                    # Only this runner's child is affected by infrastructure failure.
                    if process is not None and process.poll() is None:
                        process.terminate()
                        record["exit_code"] = process.wait()
                    process = None
            record["finished_at"] = utc_now()
            record["status"] = "failed" if record["error"] else "completed"
            state["failed_jobs" if record["error"] else "completed_jobs"].append(job["id"])
            state["current_job"], state["job_pid"] = None, None
            atomic_status(status_path, state)
        state["status"] = "completed_with_errors" if state["failed_jobs"] else "completed"
        state["finished_at"] = utc_now()
        atomic_status(status_path, state)
        return 1 if state["failed_jobs"] else 0
    except BaseException as error:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        state.update(
            status="failed",
            finished_at=utc_now(),
            job_pid=None,
            error=f"{type(error).__name__}: {error}",
        )
        atomic_status(status_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    try:
        return run_queue(args.queue)
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
