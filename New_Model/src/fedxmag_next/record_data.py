"""Deterministic label-independent sampling of actual TON-IoT connection rows.

IP identity, date, labels, IDS alerts, and host telemetry never enter the feature
matrix. Labels are stored separately for normal-only role fitting and evaluation.
Six parser processes can be used without six copies of a full dataset in memory.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

NUMERIC = (
    "duration",
    "src_bytes",
    "dst_bytes",
    "src_pkts",
    "dst_pkts",
    "src_ip_bytes",
    "dst_ip_bytes",
    "missed_bytes",
)
PROTOCOLS = ("tcp", "udp", "icmp")
SERVICES = ("dns", "http", "ssl", "ssh", "ftp", "smtp", "mqtt", "dhcp")
STATES = (
    "S0",
    "S1",
    "SF",
    "REJ",
    "S2",
    "S3",
    "RSTO",
    "RSTR",
    "RSTOS0",
    "RSTRH",
    "SH",
    "SHR",
    "OTH",
)
PORTS = (
    21,
    22,
    23,
    25,
    53,
    67,
    80,
    110,
    123,
    135,
    139,
    143,
    161,
    389,
    443,
    445,
    993,
    1883,
    3389,
    8080,
)
FEATURES = (
    *[f"log_{name}" for name in NUMERIC],
    *[f"protocol_{name}" for name in PROTOCOLS],
    "protocol_other",
    *[f"service_{name}" for name in SERVICES],
    "service_other",
    *[f"state_{name}" for name in STATES],
    "state_other",
    "log_dst_port",
    "source_privileged_port",
    "destination_privileged_port",
    *[f"destination_port_{port}" for port in PORTS],
    "destination_port_other",
    "log_http_uri_length",
    "uri_percent_fraction",
    "uri_punctuation_fraction",
    *[f"available_{name}" for name in NUMERIC],
    "available_http_uri",
    "available_src_port",
    "available_dst_port",
)
XIIOTID_COLUMNS = {
    "Timestamp": "ts",
    "Scr_IP": "src_ip",
    "Des_IP": "dst_ip",
    "Scr_port": "src_port",
    "Des_port": "dst_port",
    "Protocol": "proto",
    "Service": "service",
    "Duration": "duration",
    "Scr_bytes": "src_bytes",
    "Des_bytes": "dst_bytes",
    "Scr_pkts": "src_pkts",
    "Des_pkts": "dst_pkts",
    "Scr_ip_bytes": "src_ip_bytes",
    "Des_ip_bytes": "dst_ip_bytes",
    "Conn_state": "conn_state",
    "class3": "label",
    "class2": "type",
}


def _parse_file(task: tuple) -> dict:
    import pandas as pd

    filename, file_index, denominator = task[:3]
    options = task[3] if len(task) > 3 else {}
    extra_rows = (
        np.load(options["extra_rows"]) if options.get("extra_rows") else np.empty(0, dtype=np.int64)
    )
    usecols = [
        "ts",
        "src_ip",
        "dst_ip",
        "proto",
        "service",
        "conn_state",
        "src_port",
        "dst_port",
        *NUMERIC,
        "http_uri",
        "label",
        "type",
    ]
    blocks, timestamps, observed_at, labels, categories, signatures = [], [], [], [], [], []
    kept_rows = []
    row_counts = 0
    kept = 0
    inverse_names = {
        name: XIIOTID_COLUMNS.get(name, name) for name in pd.read_csv(filename, nrows=0).columns
    }
    for chunk in pd.read_csv(
        filename,
        usecols=lambda name: inverse_names[name] in usecols,
        dtype=str,
        chunksize=200000,
        keep_default_na=False,
    ):
        chunk = chunk.rename(columns=XIIOTID_COLUMNS)
        for name in usecols:
            if name not in chunk:
                chunk[name] = "-"
        # Stable mixed row index, chosen without inspecting any label or feature.
        row = np.arange(row_counts, row_counts + len(chunk), dtype=np.uint64)
        row_counts += len(chunk)
        mixed = row + np.uint64(file_index * 100000000 + 17)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        mixed ^= mixed >> np.uint64(31)
        selected = (mixed % denominator) == 0
        if len(extra_rows):
            selected |= np.isin(row, extra_rows)
        chosen = chunk.loc[selected].copy()
        if not len(chosen):
            continue
        ts = pd.to_numeric(chosen["ts"], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(ts)
        if not options.get("keep_self_connections", False):
            valid &= chosen["src_ip"].to_numpy() != chosen["dst_ip"].to_numpy()
        chosen, ts = chosen.loc[valid], ts[valid]
        if not len(chosen):
            continue
        kept_rows.append(row[selected][valid].astype(np.int64))
        values, availability = [], []
        for name in NUMERIC:
            column = pd.to_numeric(chosen[name], errors="coerce").to_numpy(dtype=np.float64)
            availability.append((np.isfinite(column) & (column >= 0)).astype(np.float32))
            values.append(
                np.log1p(np.maximum(np.nan_to_num(column, nan=0.0, posinf=0.0, neginf=0.0), 0))
            )
        for column, vocabulary in (
            ("proto", PROTOCOLS),
            ("service", SERVICES),
            ("conn_state", STATES),
        ):
            text = chosen[column].to_numpy()
            values.extend((text == name).astype(np.float32) for name in vocabulary)
            values.append((~np.isin(text, vocabulary)).astype(np.float32))
        src_port = pd.to_numeric(chosen["src_port"], errors="coerce").fillna(-1).to_numpy()
        dst_port = pd.to_numeric(chosen["dst_port"], errors="coerce").fillna(-1).to_numpy()
        values.extend(
            (
                np.log1p(np.maximum(dst_port, 0)),
                ((src_port >= 0) & (src_port < 1024)).astype(np.float32),
                ((dst_port >= 0) & (dst_port < 1024)).astype(np.float32),
            )
        )
        values.extend((dst_port == port).astype(np.float32) for port in PORTS)
        values.append((~np.isin(dst_port, PORTS)).astype(np.float32))
        uri = chosen["http_uri"].replace("-", "")
        length = uri.str.len().to_numpy(dtype=np.float32)
        values.extend(
            (
                np.log1p(length),
                uri.str.count("%").to_numpy() / np.maximum(length, 1),
                uri.str.count(r"[<>;'\"(){}]").to_numpy() / np.maximum(length, 1),
            )
        )
        values.extend(availability)
        values.extend(
            (
                (length > 0).astype(np.float32),
                (src_port >= 0).astype(np.float32),
                (dst_port >= 0).astype(np.float32),
            )
        )
        block = np.stack(values, axis=1).astype(np.float32)
        if not np.isfinite(block).all() or block.shape[1] != len(FEATURES):
            raise ValueError("Invalid record feature schema")
        # Hash input observations without labels/types to remove duplicates and
        # to keep identical measured connections in the same future role.
        observation = chosen.drop(columns=["label", "type"])
        signatures.append(
            pd.util.hash_pandas_object(observation, index=False).to_numpy(dtype=np.uint64)
        )
        label_text = chosen["label"].str.strip().str.casefold()
        translated = label_text.replace(
            {"normal": "0", "benign": "0", "false": "0", "attack": "1", "true": "1"}
        )
        label = pd.to_numeric(translated, errors="raise").to_numpy(dtype=np.int8)
        if not np.isin(label, [0, 1]).all():
            raise ValueError("Invalid binary record label")
        blocks.append(block)
        timestamps.append(ts.astype(np.int64))
        duration = pd.to_numeric(chosen["duration"], errors="coerce").to_numpy(dtype=np.float64)
        duration = np.maximum(np.nan_to_num(duration, nan=0.0, posinf=0.0, neginf=0.0), 0)
        observed_at.append(np.ceil(ts + duration).astype(np.int64))
        labels.append(label)
        categories.append(chosen["type"].to_numpy(dtype=str))
        kept += len(chosen)
    return {
        "file": Path(filename).name,
        "rows_read": row_counts,
        "rows_sampled": kept,
        "features": np.concatenate(blocks),
        "timestamp": np.concatenate(timestamps),
        "observed_at": np.concatenate(observed_at),
        "label": np.concatenate(labels),
        "type": np.concatenate(categories),
        "signature": np.concatenate(signatures),
        "source_row_index": np.concatenate(kept_rows),
        "file_index": file_index,
    }


def run(args: argparse.Namespace) -> None:
    from .experiments import sha256, write_json

    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.dataset == "ton":
        directory = root / "data/TON-IoT/Processed_datasets/Processed_Network_dataset"
        paths = sorted(
            directory.glob("Network_dataset_*.csv"), key=lambda p: int(p.stem.split("_")[-1])
        )
        if len(paths) != 23:
            raise ValueError("Expected the 23 TON-IoT processed network tables")
        offset = 10 * 3600
    elif args.dataset == "cic":
        directory = root / "data/processed/cic_ids2017_conn/labelled"
        paths = [directory / f"{day}.csv" for day in ("Monday", "Thursday", "Friday")]
        offset = -3 * 3600
    else:
        paths = [root / "data/X-IIoTID/X-IIoTID dataset.csv"]
        offset = 0
    write_json(
        output / "registration.json",
        {
            "status": "record_level_label_independent_preparation",
            "args": vars(args),
            "features": list(FEATURES),
            "sampling": "mixed row index plus optional unlabelled quota union",
            "quota_rows_sha256": {
                p.name: sha256(p) for p in sorted(Path(args.extra_row_directory).glob("rows_*.npy"))
            }
            if getattr(args, "extra_row_directory", None)
            else {},
            "fit_protocol": (
                "normal-only, capture-disjoint source; "
                "separate target reference/calibration prefixes"
            ),
            "causality": "Score after flow completion: observed_at = ceil(ts + duration)",
            "raw_files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in paths
            },
        },
    )
    tasks = [
        (
            str(path),
            index,
            args.denominator,
            {
                "keep_self_connections": getattr(args, "keep_self_connections", False),
                "extra_rows": str(Path(args.extra_row_directory) / f"rows_{index:02}.npy")
                if getattr(args, "extra_row_directory", None)
                else None,
            },
        )
        for index, path in enumerate(paths)
    ]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(_parse_file, tasks):
            print(
                json.dumps({key: result[key] for key in ("file", "rows_read", "rows_sampled")}),
                flush=True,
            )
            results.append(result)
    arrays = {
        key: np.concatenate([result[key] for result in results])
        for key in ("features", "timestamp", "observed_at", "label", "type", "signature")
    }
    arrays["source_row_index"] = np.concatenate([r["source_row_index"] for r in results])
    arrays["source_file_index"] = np.concatenate(
        [np.full(len(r["signature"]), r["file_index"], dtype=np.int16) for r in results]
    )
    unique, positions = np.unique(arrays["signature"], return_index=True)
    # A duplicate observation with contradictory labels cannot be silently used.
    order = np.argsort(arrays["signature"], kind="stable")
    same = arrays["signature"][order][1:] == arrays["signature"][order][:-1]
    if np.any(same & (arrays["label"][order][1:] != arrays["label"][order][:-1])):
        raise ValueError("Identical measured connection has contradictory labels")
    arrays = {key: value[positions] for key, value in arrays.items()}
    arrays["capture"] = (arrays["observed_at"] + offset) // 86400
    arrays["feature_names"] = np.asarray(FEATURES)
    arrays["dataset"] = np.asarray(args.dataset)
    arrays["utc_offset_seconds"] = np.asarray(offset)
    np.savez_compressed(output / "records.npz", **arrays)
    write_json(
        output / "report.json",
        {
            "records": len(unique),
            "normal": int(np.sum(arrays["label"] == 0)),
            "attack": int(np.sum(arrays["label"] == 1)),
            "captures": len(set(arrays["capture"].tolist())),
            "raw_rows_read": sum(result["rows_read"] for result in results),
            "selected_before_deduplication": sum(result["rows_sampled"] for result in results),
            "output_sha256": sha256(output / "records.npz"),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--denominator", type=int, default=32)
    parser.add_argument("--dataset", choices=("ton", "cic", "xiiotid"), default="ton")
    parser.add_argument("--extra-row-directory")
    parser.add_argument("--keep-self-connections", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
