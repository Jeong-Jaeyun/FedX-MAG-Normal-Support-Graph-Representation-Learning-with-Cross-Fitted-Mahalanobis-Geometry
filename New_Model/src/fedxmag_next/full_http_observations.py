"""Fill the existing full CIC query set from all actual capture HTTP observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .experiments import sha256, write_json
from .pcap_http import HTTP_FEATURES, extract

URI_COLUMNS = {
    "log_http_uri_length": 2,
    "uri_percent_fraction": 3,
    "uri_punctuation_fraction": 4,
    "available_http_uri": 1,
}


def fill_arrays(data, summaries):
    if (
        summaries.shape != (len(data["signature"]), len(HTTP_FEATURES))
        or not np.isfinite(summaries).all()
    ):
        raise ValueError("Full HTTP summaries must align with every prepared observation")
    features = data["features"].copy()
    positions = {str(name): i for i, name in enumerate(data["feature_names"])}
    for name, i in URI_COLUMNS.items():
        features[:, positions[name]] = (
            summaries[:, i] > 0 if name == "available_http_uri" else summaries[:, i]
        )
    for i, name in enumerate(HTTP_FEATURES):
        features[:, positions[f"payload_{name}"]] = summaries[:, i]
    inspected = data["http_summary_inspected"]
    if not np.array_equal(features[inspected], data["features"][inspected]):
        raise ValueError("Full packet extraction changed a previously inspected feature vector")
    allowed = set(URI_COLUMNS) | {f"payload_{name}" for name in HTTP_FEATURES}
    other = [i for name, i in positions.items() if name not in allowed]
    if not np.array_equal(features[:, other], data["features"][:, other]):
        raise ValueError("HTTP extraction changed a non-HTTP feature")
    return {
        **data,
        "features": features,
        "http_features": summaries.copy(),
        "http_feature_names": np.asarray(HTTP_FEATURES),
        "http_summary_inspected": np.ones(len(features), dtype=bool),
    }


def extract_shard(task):
    import pandas as pd

    day, file_index, capture, data_path, output_path, tracker_root, expected_md5 = task
    sys.path.insert(0, tracker_root)
    from fedxmag import pcap_conn

    if Path(pcap_conn.__file__).resolve() != Path(tracker_root, "fedxmag/pcap_conn.py").resolve():
        raise ValueError("HTTP worker did not import its frozen packet tracker")
    capture, output_path = Path(capture), Path(output_path)
    print(json.dumps({"phase": "capture_checksum", "day": day}), flush=True)
    md5, sha = hashlib.md5(), hashlib.sha256()
    with capture.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != expected_md5:
        raise ValueError(f"Capture differs from prior validated extraction: {day}")
    published = capture.with_suffix(".md5")
    if published.exists() and md5.hexdigest() != published.read_text().split()[0].lower():
        raise ValueError(f"Published capture checksum mismatch: {day}")
    with np.load(data_path) as payload:
        ids = np.flatnonzero(payload["source_file_index"] == file_index)
        signatures = payload["signature"][ids]
    wanted = {int(value): i for i, value in enumerate(signatures)}
    if len(wanted) != len(ids):
        raise ValueError("Prepared HTTP queries require unique signatures")
    extras = np.zeros((len(ids), len(HTTP_FEATURES)), dtype=np.float32)
    seen = np.zeros(len(ids), dtype=bool)

    def progress(position, size, packets):
        print(
            json.dumps(
                {
                    "phase": "http_packets",
                    "day": day,
                    "packets": packets,
                    "byte_fraction": position / size,
                }
            ),
            flush=True,
        )

    records, observed, counts = extract(capture, progress=progress)
    matched = 0
    for start in range(0, len(records), 50000):
        block = records[start : start + 50000]
        strings = [
            [
                str(r[0] // 10**9),
                str(r[1]),
                str(r[2]),
                str(r[3]),
                str(r[4]),
                r[5],
                r[6],
                f"{r[7] / 10**9:.6f}",
                str(r[8]),
                str(r[9]),
                r[10],
                str(r[11]),
                str(r[12]),
            ]
            for r in block
        ]
        frame = pd.DataFrame(strings, columns=pcap_conn.FIELDS)
        for name in ("src_ip_bytes", "dst_ip_bytes", "missed_bytes", "http_uri"):
            frame[name] = "-"
        hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
        for record, signature in zip(block, hashed, strict=True):
            index = wanted.get(int(signature))
            if index is not None:
                seen[index] = True
                matched += 1
                if record in observed:
                    extras[index] = observed[record]
    if not seen.all():
        raise ValueError(f"HTTP capture join failed for {int((~seen).sum())} rows: {day}")
    np.savez_compressed(output_path, indices=ids, signature=signatures, http_features=extras)
    report = {
        "day": day,
        "capture_md5": md5.hexdigest(),
        "capture_sha256": sha.hexdigest(),
        "capture_bytes": capture.stat().st_size,
        "reconstructed_connections": len(records),
        "matched_connection_records": matched,
        "covered_prepared_rows": len(ids),
        "http_observed_rows": int((extras[:, 0] > 0).sum()),
        "packet_counts": counts,
        "shard_sha256": sha256(output_path),
    }
    output_path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "http_shard_complete", **report}), flush=True)
    return str(output_path), report


def run(args):
    source, output, root, tracker = map(
        lambda p: Path(p).resolve(), (args.data, args.output, args.root, args.tracker_root)
    )
    source_hash = sha256(source)
    previous = json.loads(
        (root / "New_Model/artifacts/cic_connections_http_v1/report.json").read_text()
    )
    old_registration = json.loads(
        (root / "New_Model/artifacts/cic_connections_http_v1/registration.json").read_text()
    )
    tracker_hash = sha256(tracker / "fedxmag/pcap_conn.py")
    if tracker_hash != old_registration["source_tracker_sha256"]:
        raise ValueError("Frozen packet tracker differs from the validated legacy observer")
    output.mkdir(parents=True, exist_ok=False)
    packed = output / "packed"
    packed.mkdir()
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "input_sha256": source_hash,
            "source_tracker_sha256": tracker_hash,
            "schema": list(HTTP_FEATURES),
            "label_use": "none in packet extraction or signature join; metadata copied unchanged",
            "payload_limit_per_direction": 16384,
            "scope": "same existing Monday/Thursday/Friday prepared queries",
        },
    )
    tasks = [
        (
            day,
            i,
            str(root / f"data/CIC-IDS2017/{day}-WorkingHours.pcap"),
            str(source),
            str(packed / f"{day}_http.npz"),
            str(tracker),
            previous["days"][day]["md5"],
        )
        for i, day in enumerate(("Monday", "Thursday", "Friday"))
    ]
    reports = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_shard, task) for task in tasks]
        for future in as_completed(futures):
            path, report = future.result()
            reports[report["day"]] = {**report, "path": path}
    if sha256(source) != source_hash:
        raise ValueError("Prepared observation file changed during capture extraction")
    with np.load(source) as payload:
        data = {k: payload[k] for k in payload.files}
    summaries = np.zeros((len(data["signature"]), len(HTTP_FEATURES)), dtype=np.float32)
    seen = np.zeros(len(summaries), dtype=bool)
    for report in reports.values():
        with np.load(report["path"]) as shard:
            ids = shard["indices"]
            if seen[ids].any() or not np.array_equal(shard["signature"], data["signature"][ids]):
                raise ValueError("Full HTTP shards overlap or have misaligned signatures")
            summaries[ids] = shard["http_features"]
            seen[ids] = True
    if not seen.all():
        raise ValueError("Some prepared observations were never inspected")
    result = fill_arrays(data, summaries)
    np.savez_compressed(output / "records.npz", **result)
    write_json(
        output / "report.json",
        {
            "status": "actual_packet_http_for_every_prepared_query",
            "rows": len(summaries),
            "old_inspected_rows_bit_exact": int(data["http_summary_inspected"].sum()),
            "newly_inspected_rows": int((~data["http_summary_inspected"]).sum()),
            "http_observed_rows": int((summaries[:, 0] > 0).sum()),
            "all_other_features_exact": True,
            "row_identity_times_labels_types_unchanged": True,
            "captures": reports,
            "output_sha256": sha256(output / "records.npz"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracker-root", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=3)
    run(parser.parse_args())
