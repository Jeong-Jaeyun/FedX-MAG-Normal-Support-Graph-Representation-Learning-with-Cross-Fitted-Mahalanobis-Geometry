"""Convert the CIC-IDS2017 pcapng captures to label-free Zeek-style connection tables.

Each day's capture becomes ``<day>.csv`` in TON_IoT conn.log conventions
(:mod:`fedxmag.pcap_conn`) and ``<day>_manifest.json`` with the capture MD5 (checked
against the published ``.md5`` file when one exists), the code hashes, the packet and
connection counts, and the table SHA-256. No label, attack schedule, or address list
is read. Outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from fedxmag.confirmation import sha256_file
from fedxmag.pcap_conn import track_pcapng, write_conn_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/CIC-IDS2017"
OUTPUT = ROOT / "data/processed/cic_ids2017_conn"
DAYS = ("Monday", "Thursday", "Friday")
CODE = ("src/fedxmag/pcap_conn.py", "scripts/convert_cic_ids2017.py")


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def convert(day: str, data: Path, output: Path) -> None:
    capture = data / f"{day}-WorkingHours.pcap"
    table, manifest = output / f"{day}.csv", output / f"{day}_manifest.json"
    if table.exists() or manifest.exists():
        raise FileExistsError(table)
    started = time.perf_counter()
    published = data / f"{day}-WorkingHours.md5"
    expected = published.read_text().split()[0].lower() if published.exists() else None
    observed = md5_file(capture)
    if expected is not None and observed != expected:
        raise ValueError(f"{capture.name}: MD5 {observed} differs from published {expected}")
    tracker = track_pcapng(capture)
    records = tracker.close()
    output.mkdir(parents=True, exist_ok=True)
    write_conn_csv(records, table)
    starts = [record[0] for record in records]
    report = {
        "status": "label_free_conversion",
        "capture": capture.name,
        "capture_bytes": capture.stat().st_size,
        "capture_md5": observed,
        "published_md5": expected,
        "code_sha256": {name: sha256_file(ROOT / name) for name in CODE},
        "counts": dict(sorted(tracker.counts.items())),
        "conn_state": dict(Counter(record[10] for record in records).most_common()),
        "service": dict(Counter(record[6] for record in records).most_common()),
        "rows": len(records),
        "first_ts_utc": datetime.fromtimestamp(min(starts) / 1e9, UTC).isoformat(),
        "last_ts_utc": datetime.fromtimestamp(max(starts) / 1e9, UTC).isoformat(),
        "table": table.name,
        "table_sha256": sha256_file(table),
        "seconds": round(time.perf_counter() - started),
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({day: {"rows": report["rows"], "seconds": report["seconds"]}}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", choices=DAYS, action="append")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    for day in args.day or DAYS:
        convert(day, args.data, args.output)


if __name__ == "__main__":
    main()
