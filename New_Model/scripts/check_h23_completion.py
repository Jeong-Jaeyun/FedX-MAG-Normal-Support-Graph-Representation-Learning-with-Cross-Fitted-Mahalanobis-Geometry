"""Verify the final H2/H3 evidence bundle without training or changing models."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def certify(root, output):
    root = Path(root).resolve()
    output = Path(output).resolve()
    cached = {}

    def digest(path):
        path = Path(path).resolve()
        if path not in cached:
            checksum = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum.update(block)
            cached[path] = checksum.hexdigest()
        return cached[path]

    def require(value, message):
        if not value:
            raise ValueError(message)

    def read(relative):
        return json.loads((root / relative).read_text(encoding="utf-8"))

    def references(value):
        if isinstance(value, dict):
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
                require(
                    digest(value["path"]) == value["sha256"], f"Changed artifact: {value['path']}"
                )
            for item in value.values():
                references(item)
        elif isinstance(value, list):
            for item in value:
                references(item)

    queue_names = ("round234_h2_h3_v1", "round236_h23_diagnostics_v1")
    queue_evidence = {}
    for name, count in zip(queue_names, (7, 4), strict=True):
        folder = root / "New_Model/queues" / name
        state = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        require(state["status"] == "completed" and not state["failed_jobs"], name)
        require(len(state["jobs"]) == count, "Queue job coverage differs")
        require(
            digest(folder / "manifest.json") == state["manifest_sha256"], "Queue manifest changed"
        )
        for job in state["jobs"]:
            require(job["status"] == "completed" and job["exit_code"] == 0, job["id"])
            result = json.loads(Path(job["expected_report"]).read_text(encoding="utf-8"))
            require(result["status"] == "completed", f"Incomplete output: {job['id']}")
        queue_evidence[name] = {
            "jobs_completed": count,
            "finished_at": state["finished_at"],
            "status_sha256": digest(folder / "status.json"),
            "manifest_sha256": state["manifest_sha256"],
        }

    for name in (*queue_names, "round237_h23_final_audit_v2"):
        snapshot = root / "New_Model/experiment_sources" / name
        manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
        for path, expected in manifest["sha256"].items():
            require(digest(snapshot / path) == expected, f"Frozen source changed: {name}/{path}")

    real_path = "New_Model/reports/h23_full_2026-11-16_v1/report.json"
    generic_path = "New_Model/reports/h23_mechanisms_full_2026-11-17_v2/report.json"
    finding_path = "New_Model/reports/h23_findings_2026-11-16_v1/findings.json"
    real, generic, findings = [read(p) for p in (real_path, generic_path, finding_path)]
    first_snapshot = root / "New_Model/experiment_sources/round234_h2_h3_v1"
    second_snapshot = root / "New_Model/experiment_sources/round236_h23_diagnostics_v1"
    final_snapshot = root / "New_Model/experiment_sources/round237_h23_final_audit_v2"
    for actual, source in (
        (real["analysis_source_sha256"], first_snapshot / "scripts/audit_h23_results.py"),
        (real["run_manifest_sha256"], root / "New_Model/queues/round234_h2_h3_v1/runs.json"),
        (generic["audit_source_sha256"], final_snapshot / "audit_h23_mechanisms.py"),
        (generic["independent_replay_source_sha256"], final_snapshot / "audit_h23_results.py"),
        (
            generic["source_generator_sha256"],
            second_snapshot / "src/fedxmag_next/h23_mechanisms.py",
        ),
        (findings["summary_source_sha256"], second_snapshot / "scripts/summarize_h23_findings.py"),
    ):
        require(actual == digest(source), f"Report/execution-source mismatch: {source}")
    for report, expected in ((real, 522), (generic, 1152)):
        require(report["status"] == "completed" and report["smoke"] is False, "Incomplete audit")
        require(report["total_policy_conditions_checked"] == expected, "Audit scope differs")
        references(report)
    require(len(real["runs"]) == 6, "Missing real-data run")
    for run in real["runs"]:
        require(run["all_decisions_thresholds_counts_rates_exact"], "Real replay differs")
        require(run["all_source_observation_roles_disjoint"], "Real source roles overlap")
        require(run["full_target_census_matches_h1"], "Real target census differs")
    require(len(generic["tasks"]) == 21 and len(generic["domains"]) == 3, "Synthetic scope differs")
    for task in generic["tasks"].values():
        require(task["all_flags_thresholds_counts_rates_exact"], "Synthetic replay differs")
        require(
            task["negative_control_outcome_used_as_acceptance_gate"] is False, "Outcome filtering"
        )
    for domain in generic["domains"].values():
        for hypothesis in ("h2", "h3"):
            source = domain[hypothesis]
            for field in (
                "source_observation_roles_disjoint",
                "adapter_train_only_verified",
                "support_fit_budget_verified",
            ):
                require(source[field] is True, f"Synthetic source check incomplete: {field}")
            if hypothesis == "h3":
                require(source["history_observation_membership_exact"], "History IDs differ")
            require(Path(source["draws"]["path"]).name == "draws.pt", "Wrong source-data pointer")
    require(
        findings["status"] == "completed" and findings["policy_conditions"] == 522,
        "Findings incomplete",
    )
    require(findings["audit"]["sha256"] == digest(root / real_path), "Findings/audit mismatch")
    require(len(findings["practical_screens"]) == 57, "Practical comparison coverage differs")

    documents = (
        "H2_H3_RESULTS_2026-11-17.md",
        "RESEARCH_MEMORY.md",
        "LATEST_STATE.md",
        "H2_PROTOCOL_2026-11-16.md",
        "H3_PROTOCOL_2026-11-16.md",
        "H23_MECHANISMS_PROTOCOL_2026-11-16.md",
    )
    document_evidence = {name: digest(root / "New_Model/docs" / name) for name in documents}
    checker_hash = digest(__file__)
    result = {
        "status": "completed",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "objective": "Complete registered H2/H3 research experiments and mechanism diagnostics",
        "real_full_runs": 6,
        "synthetic_tasks": 21,
        "policy_conditions_checked": 1674,
        "queues": queue_evidence,
        "unique_files_rehashed": len(cached),
        "real_audit": {"path": str(root / real_path), "sha256": digest(root / real_path)},
        "synthetic_final_audit": {
            "path": str(root / generic_path),
            "sha256": digest(root / generic_path),
        },
        "documents_sha256": document_evidence,
        "practical_screens_passed": sum(
            v["passed"] for v in findings["practical_screens"].values()
        ),
        "conditional_expansion": "not triggered: common real-data development gates failed",
        "claims_excluded": [
            "all-three95/5 achieved",
            "independent real-site generalization",
            "QPU/finite-shot accuracy",
            "computational quantum advantage",
        ],
        "scientific_verdict": (
            "configuration-specific negative results with the documented X scalar exception"
        ),
        "checker_source_sha256": checker_hash,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "real_full_runs",
                    "synthetic_tasks",
                    "policy_conditions_checked",
                    "unique_files_rehashed",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    certify(args.root, args.output)
