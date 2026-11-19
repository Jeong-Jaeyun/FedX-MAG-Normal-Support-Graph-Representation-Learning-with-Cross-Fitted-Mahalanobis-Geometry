"""Wait for an upstream queue to release its CPU workers; never stop its jobs.

Usage: python wait_research_queue.py --upstream C:/queue/manifest.json --output C:/new/gate

A completed gate report means that the identified upstream queue and running
children are no longer alive. It does not assert that upstream experiments
succeeded. An independent downstream experiment may proceed after an upstream
failure. Unreadable state or an unqueryable process fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

TERMINAL = {"completed", "completed_with_errors", "failed"}


def utc_now():
    return datetime.now(UTC).isoformat()


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("A process with a recorded PID requires its start timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Process timestamps must include a timezone")
    return result.timestamp()


def windows_process_info(pid):
    """Query a process without executing a shell or requiring optional packages."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        error = ctypes.get_last_error()
        if error in (87, 1168):  # invalid PID / not found
            return None
        raise OSError(error, f"Cannot query process {pid}")
    try:
        creation, exit_time, kernel_time, user_time = [wintypes.FILETIME() for _ in range(4)]
        exit_code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        created = ((creation.dwHighDateTime << 32) | creation.dwLowDateTime) / 10_000_000
        exited = (exit_time.dwHighDateTime << 32) | exit_time.dwLowDateTime
        if exited or exit_code.value != 259:
            return {"alive": False, "created_at": created - 11_644_473_600, "image": None}
        size = wintypes.DWORD(32768)
        image = ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"alive": True, "created_at": created - 11_644_473_600, "image": image.value}
    finally:
        kernel.CloseHandle(handle)


def process_info(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
        raise ValueError("Process IDs must be positive integers")
    if os.name == "nt":
        return windows_process_info(pid)
    if sys.platform.startswith("linux"):
        # Linux support keeps the same birth-time/image contract for local tests.
        try:
            process = Path(f"/proc/{pid}")
            stat = (process / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            if fields[0] == "Z":
                return {"alive": False, "created_at": None, "image": None}
            boot = next(
                int(line.split()[1])
                for line in Path("/proc/stat").read_text().splitlines()
                if line.startswith("btime ")
            )
            created = boot + int(fields[19]) / os.sysconf("SC_CLK_TCK")
            return {
                "alive": True,
                "created_at": created,
                "image": str((process / "exe").resolve(strict=True)),
            }
        except (FileNotFoundError, ProcessLookupError):
            return None
    raise RuntimeError("Authoritative process birth-time queries require Windows or Linux")


def identity(pid, started_at, python, probe):
    info = probe(pid)
    if info is None or not info.get("alive"):
        return {"pid": pid, "live": False, "reason": "process_missing_or_exited"}
    if not isinstance(info.get("created_at"), (int, float)) or not isinstance(
        info.get("image"), str
    ):
        raise ValueError(f"Process {pid} identity could not be verified")
    if abs(info["created_at"] - timestamp(started_at)) > 10:
        return {"pid": pid, "live": False, "reason": "pid_reused_birth_time_differs", **info}
    if os.path.normcase(os.path.realpath(info["image"])) != os.path.normcase(
        os.path.realpath(python)
    ):
        return {"pid": pid, "live": False, "reason": "pid_reused_image_differs", **info}
    return {"pid": pid, "live": True, "reason": "birth_and_image_match", **info}


def read_upstream(manifest_path, manifest, digest):
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != digest:
        raise ValueError("Upstream manifest changed while waiting")
    state = json.loads((manifest_path.parent / "status.json").read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise ValueError("Unrecognized upstream queue state")
    if state.get("manifest_sha256") != digest:
        raise ValueError("Upstream manifest hash differs from its queue state")
    if Path(state.get("manifest", "")).resolve() != manifest_path:
        raise ValueError("Upstream state points to another manifest")
    if state.get("status") not in TERMINAL | {"running"}:
        raise ValueError("Unknown upstream queue status")
    expected = [j["id"] for j in manifest["jobs"]]
    if [j.get("id") for j in state.get("jobs", [])] != expected:
        raise ValueError("Upstream job list differs from its manifest")
    timestamp(state.get("started_at"))
    if state.get("current_job") is not None and state["current_job"] not in expected:
        raise ValueError("Unknown current upstream job")
    return state


def atomic_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_gate(upstream, output, poll_seconds=10.0, *, probe=process_info, sleep=time.sleep):
    upstream, output = Path(upstream), Path(output)
    if not upstream.is_absolute() or not output.is_absolute() or poll_seconds <= 0:
        raise ValueError("Absolute upstream/output paths and a positive poll interval are required")
    upstream = upstream.resolve()
    raw = upstream.read_bytes()
    manifest = json.loads(raw)
    for field in ("root", "python"):
        if not isinstance(manifest.get(field), str) or not Path(manifest[field]).is_absolute():
            raise ValueError(f"Manifest {field} must be absolute")
    if not Path(manifest["root"]).is_dir() or not Path(manifest["python"]).is_file():
        raise ValueError("Upstream workspace or Python does not exist")
    if not isinstance(manifest.get("jobs"), list) or not manifest["jobs"]:
        raise ValueError("Upstream jobs must be a nonempty list")
    job_ids = [j.get("id") for j in manifest["jobs"]]
    if not all(isinstance(i, str) and i for i in job_ids) or len(set(job_ids)) != len(job_ids):
        raise ValueError("Upstream job identifiers must be unique strings")
    digest = hashlib.sha256(raw).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    started_at, observations, previous = utc_now(), 0, None
    try:
        while True:
            state = read_upstream(upstream, manifest, digest)
            queue = identity(state.get("pid"), state["started_at"], manifest["python"], probe)
            running = [j for j in state["jobs"] if j.get("status") == "running"]
            current = next((j for j in state["jobs"] if j["id"] == state.get("current_job")), None)
            if state.get("job_pid") is not None:
                if current is None or current.get("pid") != state["job_pid"]:
                    raise ValueError("Current child PID is not identified by its job record")
                if current not in running:
                    running.append(current)
            children = []
            for job in running:
                if job.get("pid") is None:
                    if not queue["live"]:
                        raise ValueError("Dead queue has an unidentified possibly-started child")
                    continue
                children.append(
                    {
                        "job": job["id"],
                        **identity(job["pid"], job.get("started_at"), manifest["python"], probe),
                    }
                )
            observations += 1
            observed = {
                "status": "waiting",
                "pid": os.getpid(),
                "started_at": started_at,
                "heartbeat_at": utc_now(),
                "upstream_manifest": str(upstream),
                "upstream_manifest_sha256": digest,
                "upstream_status": state["status"],
                "queue_process": queue,
                "child_processes": children,
                "observations": observations,
            }
            busy = queue["live"] or any(child["live"] for child in children)
            if not busy:
                interrupted = state["status"] not in TERMINAL
                observed.update(
                    status="completed",
                    finished_at=utc_now(),
                    resource_gate_passed=True,
                    upstream_interrupted=interrupted,
                    upstream_experiments_succeeded=state["status"] == "completed",
                    scope="CPU worker release only; not an upstream experiment success assertion",
                )
                with (output / "report.json").open("x", encoding="utf-8") as handle:
                    json.dump(observed, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                atomic_json(output / "wait_status.json", observed)
                print("UPSTREAM WORKERS RELEASED", state["status"], flush=True)
                return observed
            atomic_json(output / "wait_status.json", observed)
            signature = (
                state["status"],
                queue["live"],
                tuple((c["job"], c["pid"], c["live"]) for c in children),
            )
            if signature != previous:
                print("WAITING", json.dumps({"queue": queue, "children": children}), flush=True)
                previous = signature
            sleep(poll_seconds)
    except BaseException as error:
        atomic_json(
            output / "failure.json",
            {
                "status": "failed",
                "started_at": started_at,
                "finished_at": utc_now(),
                "error": f"{type(error).__name__}: {error}",
                "resource_gate_passed": False,
                "upstream_manifest": str(upstream),
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_gate(args.upstream, args.output)


if __name__ == "__main__":
    main()
