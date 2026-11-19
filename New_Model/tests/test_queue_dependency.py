"""Prove the dependency gate waits for actual workers and rejects uncertain state."""

import hashlib
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/wait_research_queue.py"
SPEC = importlib.util.spec_from_file_location("wait_research_queue", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

START = "2026-11-17T05:34:50+00:00"
BIRTH = datetime.fromisoformat(START).timestamp()


def fixture(tmp_path, status="running", queue_pid=101, child_pid=102):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"root": str(tmp_path), "python": sys.executable, "jobs": [{"id": "model"}]}),
        encoding="utf-8",
    )
    state = {
        "format_version": 1,
        "status": status,
        "pid": queue_pid,
        "started_at": START,
        "manifest": str(path),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "current_job": "model" if child_pid is not None else None,
        "job_pid": child_pid,
        "jobs": [
            {
                "id": "model",
                "status": "running" if child_pid is not None else "completed",
                "pid": child_pid,
                "started_at": START,
            }
        ],
    }
    save(tmp_path, state)
    return path, state


def save(tmp_path, state):
    (tmp_path / "status.json").write_text(json.dumps(state), encoding="utf-8")


def live(created=BIRTH, image=None):
    return {"alive": True, "created_at": created, "image": image or sys.executable}


def test_waits_for_dead_queue_orphan_child_then_releases(tmp_path):
    path, state = fixture(tmp_path)
    ticks = []
    original = path.read_bytes()

    def probe(pid):
        return live() if pid == 102 and not ticks else None

    def sleep(seconds):
        assert seconds == 10
        ticks.append(1)

    result = gate.run_gate(path, tmp_path / "gate", probe=probe, sleep=sleep)
    assert ticks == [1]
    assert result["resource_gate_passed"] and result["upstream_interrupted"]
    assert not result["upstream_experiments_succeeded"]
    assert path.read_bytes() == original
    assert json.loads((tmp_path / "status.json").read_text()) == state


def test_terminal_status_does_not_override_live_queue(tmp_path):
    path, state = fixture(tmp_path, status="completed", child_pid=None)
    ticks = []

    def probe(pid):
        return live() if not ticks else None

    result = gate.run_gate(path, tmp_path / "gate", probe=probe, sleep=lambda _: ticks.append(1))
    assert ticks == [1]
    assert result["upstream_experiments_succeeded"]
    assert not result["upstream_interrupted"]
    assert result["observations"] == 2


def test_reused_pid_not_treated_as_upstream_worker(tmp_path):
    path, _ = fixture(tmp_path, status="completed_with_errors", child_pid=None)
    result = gate.run_gate(
        path,
        tmp_path / "gate",
        probe=lambda _: live(created=BIRTH + 60),
        sleep=lambda _: pytest.fail("PID reuse should not wait"),
    )
    assert result["queue_process"]["reason"] == "pid_reused_birth_time_differs"
    assert result["resource_gate_passed"]
    assert not result["upstream_experiments_succeeded"]


def test_unreadable_state_fails_closed(tmp_path):
    path, _ = fixture(tmp_path)
    (tmp_path / "status.json").write_text("broken", encoding="utf-8")
    with pytest.raises(ValueError):
        gate.run_gate(path, tmp_path / "gate", probe=lambda _: pytest.fail("Must not probe"))
    assert not (tmp_path / "gate/report.json").exists()
    assert json.loads((tmp_path / "gate/failure.json").read_text())["resource_gate_passed"] is False


def test_unknown_process_state_fails_closed(tmp_path):
    path, _ = fixture(tmp_path)

    def denied(_):
        raise PermissionError("process identity unavailable")

    with pytest.raises(PermissionError):
        gate.run_gate(path, tmp_path / "gate", probe=denied)
    assert not (tmp_path / "gate/report.json").exists()


def test_changed_manifest_while_waiting_fails_closed(tmp_path):
    path, _ = fixture(tmp_path)

    def change_manifest(_):
        path.write_text(path.read_text() + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="changed while waiting"):
        gate.run_gate(path, tmp_path / "gate", probe=lambda _: live(), sleep=change_manifest)
    assert not (tmp_path / "gate/report.json").exists()


def test_manifest_hash_change_is_rejected(tmp_path):
    path, state = fixture(tmp_path)
    state["manifest_sha256"] = "different"
    save(tmp_path, state)
    with pytest.raises(ValueError, match="hash"):
        gate.run_gate(path, tmp_path / "gate", probe=lambda _: None)


def test_unidentified_child_after_dead_queue_fails_closed(tmp_path):
    path, state = fixture(tmp_path)
    state["jobs"][0]["pid"] = None
    state["job_pid"] = None
    save(tmp_path, state)
    with pytest.raises(ValueError, match="unidentified"):
        gate.run_gate(path, tmp_path / "gate", probe=lambda _: None)


def test_existing_output_not_overwritten(tmp_path):
    path, _ = fixture(tmp_path, status="completed", child_pid=None)
    output = tmp_path / "gate"
    gate.run_gate(path, output, probe=lambda _: None)
    original = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        gate.run_gate(path, output, probe=lambda _: None)
    assert (output / "report.json").read_bytes() == original


def test_actual_self_process_birth_and_image():
    observed = gate.process_info(os.getpid())
    assert observed["alive"]
    assert abs(observed["created_at"] - datetime.now(UTC).timestamp()) < 3600
    assert os.path.samefile(observed["image"], sys.executable)
    assert gate.process_info(99999999) is None
