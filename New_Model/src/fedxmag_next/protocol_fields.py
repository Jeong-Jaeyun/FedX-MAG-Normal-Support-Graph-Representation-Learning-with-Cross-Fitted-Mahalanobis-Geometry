"""Shape-only protocol metadata omitted by the common73-field connection schema.

Encode observed ICMP/source-port categories, DNS shapes/flags, TLS versions and
handshake flags, and HTTP method/status/body lengths. No IP, timestamp, label,
attack type, alert, literal domain, certificate name or user-agent enters output.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .experiments import sha256, write_json
from .record_data import FEATURES, PORTS
from .record_experiment import corruption_groups
from .temporal_context import CONTEXT_FEATURES

ICMP_VALUES = (0, 3, 5, 8, 11, 12, 128, 129, 133, 134, 135, 136, 143)
DNS_TYPES = (1, 2, 5, 12, 15, 16, 28)
DNS_RCODES = (0, 2, 3, 5)
TLS_VERSIONS = ("tlsv10", "tlsv11", "tlsv12", "tlsv13", "ssl_old", "other")
RAW_COLUMNS = (
    "proto",
    "service",
    "src_port",
    "dst_port",
    "dns_query",
    "dns_qtype",
    "dns_rcode",
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rejected",
    "ssl_version",
    "ssl_resumed",
    "ssl_established",
    "ssl_subject",
    "ssl_issuer",
    "http_method",
    "http_status_code",
    "http_request_body_len",
    "http_response_body_len",
)


def _text(frame, name):
    return frame[name].fillna("").astype(str).str.strip()


def _present(values):
    return ~values.str.lower().isin(("", "-", "?", "nan", "none", "null"))


def _numeric(frame, name, active, maximum=None):
    values = pd.to_numeric(_text(frame, name), errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(values) & (values >= 0) & np.asarray(active)
    if maximum is not None:
        valid &= (values <= maximum) & (values == np.floor(values))
    return np.where(valid, values, 0), valid


def _entropy(value):
    if not value:
        return 0.0
    counts = Counter(value)
    n = len(value)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def encode(frame):
    """Pure allowed-column transform; returned order defines the extension schema."""
    result = {}
    proto = _text(frame, "proto").str.lower()
    service = _text(frame, "service").str.lower()
    transport = proto.isin(("tcp", "udp")).to_numpy()
    icmp = proto.isin(("icmp", "icmp6", "ipv6-icmp")).to_numpy()
    source, valid = _numeric(frame, "src_port", transport, 65535)
    for port in PORTS:
        result[f"control_source_port_{port}"] = valid & (source == port)
    result["control_source_port_other"] = valid & ~np.isin(source, PORTS)
    result["control_source_port_available"] = valid
    source, valid = _numeric(frame, "src_port", icmp, 255)
    for value in ICMP_VALUES:
        result[f"control_icmp_source_{value}"] = valid & (source == value)
    result["control_icmp_source_other"] = valid & ~np.isin(source, ICMP_VALUES)
    result["control_icmp_source_available"] = valid
    # ICMP fields are Zeek pseudo-port observations; no packet-code inference.
    query = _text(frame, "dns_query")
    qpresent = _present(query).to_numpy()
    query = query.where(qpresent, "")
    length = query.str.len().to_numpy(dtype=np.float64)
    result["payload_dns_log_query_length"] = np.log1p(length)
    result["payload_dns_log_query_labels"] = np.log1p(
        query.map(lambda s: len(s.rstrip(".").split(".")) if s else 0).to_numpy(dtype=np.float64)
    )
    result["payload_dns_query_digit_fraction"] = query.str.count(r"[0-9]").to_numpy() / np.maximum(
        length, 1
    )
    result["payload_dns_query_entropy"] = query.map(_entropy).to_numpy(dtype=np.float64)
    result["payload_dns_query_available"] = qpresent
    dns = service.str.contains("dns", regex=False).to_numpy() | qpresent
    for field, values in (("dns_qtype", DNS_TYPES), ("dns_rcode", DNS_RCODES)):
        number, known = _numeric(frame, field, dns, 65535)
        for value in values:
            result[f"payload_{field}_{value}"] = known & (number == value)
        result[f"payload_{field}_other"] = known & ~np.isin(number, values)
        result[f"payload_{field}_available"] = known
    for field in ("dns_AA", "dns_RD", "dns_RA", "dns_rejected"):
        text = _text(frame, field).str.lower()
        available = text.isin(("t", "f", "true", "false", "1", "0")).to_numpy() & dns
        result[f"payload_{field}"] = text.isin(("t", "true", "1")).to_numpy() & available
        result[f"payload_{field}_available"] = available
    version = _text(frame, "ssl_version").str.lower().str.replace(".", "", regex=False)
    tls = service.str.contains("ssl", regex=False).to_numpy() | _present(version).to_numpy()
    vknown = _present(version).to_numpy() & tls
    version = version.where(~version.isin(("sslv2", "sslv3")), "ssl_old")
    version = version.where(version.isin(TLS_VERSIONS[:-1]), "other")
    for value in TLS_VERSIONS:
        result[f"payload_tls_version_{value}"] = (version == value).to_numpy() & vknown
    result["payload_tls_version_available"] = vknown
    for field in ("ssl_resumed", "ssl_established"):
        text = _text(frame, field).str.lower()
        available = text.isin(("t", "f", "true", "false", "1", "0")).to_numpy() & tls
        result[f"payload_{field}"] = text.isin(("t", "true", "1")).to_numpy() & available
        result[f"payload_{field}_available"] = available
    subject, issuer = _text(frame, "ssl_subject"), _text(frame, "ssl_issuer")
    both = (_present(subject) & _present(issuer)).to_numpy() & tls
    result["payload_tls_subject_equals_issuer"] = (subject == issuer).to_numpy() & both
    result["payload_tls_subject_issuer_available"] = both
    method = _text(frame, "http_method").str.upper()
    method_present = _present(method).to_numpy()
    http = service.str.contains("http", regex=False).to_numpy() | method_present
    for value in ("GET", "POST", "HEAD"):
        result[f"payload_http_method_{value}"] = (method == value).to_numpy() & method_present
    result["payload_http_method_other"] = (
        ~method.isin(("GET", "POST", "HEAD")).to_numpy() & method_present
    )
    result["payload_http_method_available"] = method_present
    status, known = _numeric(frame, "http_status_code", http, 999)
    known &= status > 0
    for family in range(1, 6):
        result[f"payload_http_status_{family}xx"] = known & (status // 100 == family)
    result["payload_http_status_other"] = known & ~np.isin(status // 100, range(1, 6))
    result["payload_http_status_available"] = known
    for name in ("http_request_body_len", "http_response_body_len"):
        number, known = _numeric(frame, name, http)
        result[f"payload_log_{name}"] = np.log1p(number)
        result[f"payload_{name}_available"] = known
    names = tuple(result)
    values = np.column_stack(list(result.values())).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite protocol feature")
    return names, values


def parse_file(task):
    filename, positions, raw_rows = task
    matrix = []
    offset = 0
    for frame in pd.read_csv(
        filename, usecols=list(RAW_COLUMNS), dtype=str, keep_default_na=False, chunksize=200000
    ):
        lo, hi = np.searchsorted(raw_rows, [offset, offset + len(frame)])
        if hi > lo:
            _, values = encode(frame.iloc[raw_rows[lo:hi] - offset])
            matrix.append(values)
        offset += len(frame)
    if len(raw_rows) and raw_rows[-1] >= offset:
        raise ValueError("Protocol row provenance exceeds original table")
    return filename, positions, np.concatenate(matrix)


def run(args):
    base, output = Path(args.data).resolve(), Path(args.output).resolve()
    with np.load(base) as payload:
        data = {key: payload[key] for key in payload.files}
    if str(data["dataset"]) != "ton" or tuple(data["feature_names"]) != FEATURES + CONTEXT_FEATURES:
        raise ValueError("Protocol extension expects the unchanged TON common+context schema")
    root = Path(args.root) / "data/TON-IoT/Processed_datasets/Processed_Network_dataset"
    paths = sorted(root.glob("Network_dataset_*.csv"), key=lambda p: int(p.stem.split("_")[-1]))
    if len(paths) != 23:
        raise ValueError("Expected all23 original TON network tables")
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": "ton",
            "input_sha256": sha256(base),
            "raw_sources_sha256": {p.name: sha256(p) for p in paths},
            "encoding": "allowed header/application columns only; no literal identities or labels",
            "row_selection": "exact existing prepared row provenance; no selection change",
        },
    )
    empty = pd.DataFrame({name: ["-"] for name in RAW_COLUMNS})
    names, _ = encode(empty)
    extra = np.zeros((len(data["features"]), len(names)), dtype=np.float32)
    seen = np.zeros(len(extra), dtype=bool)
    tasks = []
    for index, path in enumerate(paths):
        positions = np.flatnonzero(data["source_file_index"] == index)
        positions = positions[np.argsort(data["source_row_index"][positions])]
        if len(positions):
            tasks.append((str(path), positions, data["source_row_index"][positions]))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for filename, positions, values in pool.map(parse_file, tasks):
            if seen[positions].any() or len(positions) != len(values):
                raise ValueError("Protocol join overlaps or has missing rows")
            extra[positions] = values
            seen[positions] = True
            print({"file": Path(filename).name, "protocol_rows": len(positions)}, flush=True)
    if not seen.all():
        raise ValueError("Protocol join omitted prepared observations")
    old = data["features"]
    data["features"] = np.concatenate(
        (old[:, : len(FEATURES)], extra, old[:, len(FEATURES) :]), axis=1
    )
    data["feature_names"] = np.array(FEATURES + names + CONTEXT_FEATURES)
    corruption_groups(data["feature_names"])
    keep = np.r_[
        np.arange(len(FEATURES)), np.arange(len(FEATURES) + len(names), data["features"].shape[1])
    ]
    if not np.array_equal(data["features"][:, keep], old):
        raise ValueError("Adding protocol observations changed existing measured features")
    np.savez_compressed(output / "records.npz", **data)
    write_json(
        output / "report.json",
        {
            "records": len(extra),
            "added_dimensions": len(names),
            "dimensions": data["features"].shape[1],
            "all_existing_features_exact": True,
            "no_observation_selection_change": True,
            "output_sha256": sha256(output / "records.npz"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())
