"""Exercise real child processes, terminal report checks, and refusal to rerun."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/background_research_queue.py"
SPEC = importlib.util.spec_from_file_location("background_research_queue", SCRIPT)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def manifest(tmp_path, jobs):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "root": str(tmp_path),
                "python": sys.executable,
                "environment": {"H23_QUEUE_TEST": "expected"},
                "jobs": jobs,
            }
        ),
        encoding="utf-8",
    )
    return path


def child(identifier, report, code):
    return {"id": identifier, "argv": ["-u", "-c", code], "expected_report": str(report)}


def successful_code(report, marker, tag):
    return (
        "import json,os;from pathlib import Path;"
        "assert os.environ['H23_QUEUE_TEST']=='expected';"
        f"p=Path({str(marker)!r});p.write_text((p.read_text() if p.exists() else '')+{tag!r});"
        f"Path({str(report)!r}).write_text(json.dumps({{'status':'completed'}}));"
        f"print({tag!r})"
    )


def test_failed_job_does_not_block_later_job_and_no_rerun(tmp_path):
    first, last, missing = [tmp_path / n for n in ("first.json", "last.json", "missing.json")]
    marker = tmp_path / "sequence.txt"
    path = manifest(
        tmp_path,
        [
            child("first", first, successful_code(first, marker, "A")),
            child("bad", missing, "import sys;print('failure output');sys.exit(7)"),
            child("last", last, successful_code(last, marker, "B")),
        ],
    )
    original = path.read_bytes()
    assert queue.run_queue(path, heartbeat_seconds=0.02) == 1
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["status"] == "completed_with_errors"
    assert state["completed_jobs"] == ["first", "last"]
    assert state["failed_jobs"] == ["bad"]
    assert state["jobs"][1]["exit_code"] == 7
    assert state["current_job"] is None and state["job_pid"] is None
    assert marker.read_text() == "AB"
    assert (tmp_path / "logs/bad.log").read_text().strip() == "failure output"
    assert all(j["pid"] and j["started_at"] <= j["finished_at"] for j in state["jobs"])
    assert all(
        a["finished_at"] <= b["started_at"]
        for a, b in zip(state["jobs"][:-1], state["jobs"][1:], strict=True)
    )
    assert path.read_bytes() == original
    before = (tmp_path / "status.json").read_bytes()
    with pytest.raises(FileExistsError):
        queue.run_queue(path)
    assert (tmp_path / "status.json").read_bytes() == before


def test_zero_exit_requires_completed_report(tmp_path):
    missing, wrong = tmp_path / "missing.json", tmp_path / "wrong.json"
    path = manifest(
        tmp_path,
        [
            child("missing", missing, "print('no report')"),
            child(
                "wrong", wrong, f"from pathlib import Path;Path({str(wrong)!r}).write_text('{{}}')"
            ),
        ],
    )
    assert queue.run_queue(path, heartbeat_seconds=0.02) == 1
    state = json.loads((tmp_path / "status.json").read_text())
    assert [j["error"] for j in state["jobs"]] == [
        "expected_report_missing",
        "expected_report_not_completed",
    ]


def test_success_requires_actual_completed_report(tmp_path):
    report, marker = tmp_path / "report.json", tmp_path / "marker"
    path = manifest(tmp_path, [child("ok", report, successful_code(report, marker, "ok"))])
    assert queue.run_queue(path, heartbeat_seconds=0.02) == 0
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["status"] == "completed" and state["completed_jobs"] == ["ok"]
    assert state["finished_at"] is not None


def test_existing_report_is_not_reexecuted(tmp_path):
    report = tmp_path / "report.json"
    report.write_text('{"status":"completed"}')
    path = manifest(tmp_path, [child("old", report, "raise RuntimeError('must not launch')")])
    assert queue.run_queue(path, heartbeat_seconds=0.02) == 1
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["jobs"][0]["pid"] is None
    assert "refusing rerun" in state["jobs"][0]["error"]
    assert report.read_text() == '{"status":"completed"}'


def test_duplicate_or_unsafe_job_id_rejected_before_lock(tmp_path):
    path = manifest(tmp_path, [child("../unsafe", tmp_path / "r.json", "pass")])
    with pytest.raises(ValueError, match="Job id"):
        queue.run_queue(path)
    assert not (tmp_path / "queue.lock").exists()


@pytest.mark.parametrize("failure_code", ["raise SystemExit(7)", "print('no completed report')"])
def test_failed_gate_skips_dependents_but_independent_job_runs(tmp_path, failure_code):
    gate_report, heavy_report, last_report = [
        tmp_path / name for name in ("gate.json", "heavy.json", "last.json")
    ]
    heavy_marker, independent_marker = tmp_path / "heavy-marker", tmp_path / "independent-marker"
    gate_job = child("gate", gate_report, failure_code)
    heavy_job = child(
        "heavy", heavy_report, successful_code(heavy_report, heavy_marker, "must not run")
    )
    heavy_job["requires"] = ["gate"]
    path = manifest(
        tmp_path,
        [
            gate_job,
            heavy_job,
            child(
                "independent", last_report, successful_code(last_report, independent_marker, "ran")
            ),
        ],
    )
    assert queue.run_queue(path, heartbeat_seconds=0.02) == 1
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["status"] == "completed_with_errors"
    assert state["completed_jobs"] == ["independent"]
    assert state["failed_jobs"] == ["gate", "heavy"]
    assert state["skipped_jobs"] == ["heavy"]
    skipped = state["jobs"][1]
    assert skipped["status"] == "skipped" and skipped["pid"] is None
    assert skipped["exit_code"] is None and skipped["started_at"] is None
    assert skipped["finished_at"] and "dependency_not_completed" in skipped["error"]
    assert not heavy_marker.exists() and not heavy_report.exists()
    assert independent_marker.read_text() == "ran"


def test_successful_dependency_chain_executes_after_completed_reports(tmp_path):
    reports = [tmp_path / (name + ".json") for name in ("gate", "heavy", "audit")]
    marker = tmp_path / "order"
    jobs = [
        child(name, report, successful_code(report, marker, letter))
        for name, report, letter in zip(("gate", "heavy", "audit"), reports, "ABC", strict=True)
    ]
    jobs[1]["requires"] = ["gate"]
    jobs[2]["requires"] = ["gate", "heavy"]
    path = manifest(tmp_path, jobs)
    assert queue.run_queue(path, heartbeat_seconds=0.02) == 0
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["completed_jobs"] == ["gate", "heavy", "audit"]
    assert not state["failed_jobs"] and not state["skipped_jobs"]
    assert marker.read_text() == "ABC"


@pytest.mark.parametrize(
    "requirements", [["later"], ["first"], ["missing"], [""], "first", [None], ["known", "known"]]
)
def test_invalid_dependencies_rejected_before_lock(tmp_path, requirements):
    known = child("known", tmp_path / "known.json", "pass")
    first = child("first", tmp_path / "first.json", "pass")
    first["requires"] = requirements
    later = child("later", tmp_path / "later.json", "pass")
    path = manifest(tmp_path, [known, first, later])
    with pytest.raises(ValueError, match="requires"):
        queue.run_queue(path)
    assert not (tmp_path / "queue.lock").exists()


def test_skipped_dependency_prevents_transitive_child(tmp_path):
    marker = tmp_path / "must-not-exist"
    jobs = [child("gate", tmp_path / "gate.json", "raise SystemExit(1)")]
    for name, required in (("heavy", "gate"), ("audit", "heavy")):
        report = tmp_path / (name + ".json")
        job = child(name, report, successful_code(report, marker, "bad"))
        job["requires"] = [required]
        jobs.append(job)
    path = manifest(tmp_path, jobs)
    assert queue.run_queue(path, heartbeat_seconds=0.02) == 1
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["skipped_jobs"] == ["heavy", "audit"]
    assert not marker.exists()
