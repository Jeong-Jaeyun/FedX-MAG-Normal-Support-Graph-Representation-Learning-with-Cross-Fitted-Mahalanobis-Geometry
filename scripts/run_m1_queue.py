"""Run every registered M1 (arm, seed) training job with a bounded worker pool.

Each job is an independent ``scripts/run_m1.py train`` process with one thread,
so a failure in one job never affects another and every job keeps its own log.
Jobs whose checkpoint already exists are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "configs" / "m1_protocol.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "output" / "m1" / "xiiotid_stage1_m1"
    )
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    logs = args.output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [
        (arm, int(seed))
        for arm in protocol["arms"]
        for seed in protocol["training"]["seeds"]
        if not (args.output_dir / f"m1_{arm}_seed_{seed}.pt").exists()
    ]
    running: list[tuple[subprocess.Popen, str, object]] = []
    failures: list[str] = []
    while jobs or running:
        while jobs and len(running) < args.workers:
            arm, seed = jobs.pop(0)
            name = f"{arm}_seed_{seed}"
            log = (logs / f"{name}.log").open("w", encoding="utf-8")
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_m1.py"),
                "train",
                "--protocol",
                str(args.protocol),
                "--output-dir",
                str(args.output_dir),
                "--arm",
                arm,
                "--seed",
                str(seed),
                "--threads",
                "1",
            ]
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT
            )
            running.append((process, name, log))
            print(f"started {name}", flush=True)
        time.sleep(5)
        for entry in list(running):
            process, name, log = entry
            if process.poll() is None:
                continue
            log.close()
            running.remove(entry)
            status = "ok" if process.returncode == 0 else f"FAILED ({process.returncode})"
            if process.returncode != 0:
                failures.append(name)
            print(f"finished {name}: {status}", flush=True)
    print(f"all jobs done; failures: {failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
