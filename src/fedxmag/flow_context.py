"""Per-edge flow context that the v3b graph schema discards (protocol, flags, ports).

Each directed observed (source, destination) pair in a 60 s window becomes one
17-value vector: log record count, protocol/service mix, TCP flag rates, log mean
duration, log distinct destination ports, privileged-port share, and log mean
source/destination flow bytes. Synthetic reverse observations are not created.
Rows follow the preprocessing ingestion rules (exact-row deduplication, timestamp
parsing, endpoint validation). Labels, attack metadata, IDS alerts, and host
telemetry are never read, so unlabeled deployment captures can be featurized.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from fedxmag.preprocessing.events import _clean, _timestamp_ns
from fedxmag.schema import GraphSample

WINDOW_NS = 60 * 1_000_000_000
FLAGS = ("is_syn_only", "FIN or RST", "is_with_payload", "is_pure_ack", "Is_SYN_ACK")
NUMERIC_COLUMNS = (*FLAGS, "Duration", "Des_port", "Scr_bytes", "Des_bytes")
FLOW_FEATURES = (
    "log_records", "tcp", "udp", "icmp", "svc_dns", "svc_http", "svc_other",
    *FLAGS, "log_duration", "log_dst_ports", "privileged_port", "log_src_bytes",
    "log_dst_bytes",
)
FLOW_DIM = len(FLOW_FEATURES)
# Log record count, duration, distinct ports, and per-flow bytes scale with how busy
# an environment is; the other columns are shares and rates.
ABSOLUTE_SCALE_COLUMNS = (0, 12, 13, 15, 16)

WindowKey = tuple[str, int]


def _all(flows: Tensor) -> Tensor:
    return flows


def _activity_flags(flows: Tensor) -> Tensor:
    return flows[:, [0, *range(7, 13), 15, 16]]


def _composition(flows: Tensor) -> Tensor:
    return flows[:, [*range(1, 12), 14]]


def _window_relative(flows: Tensor) -> Tensor:
    """Centre absolute-scale columns on the window's median edge (lower median)."""

    relative = flows.clone()
    if len(flows):
        columns = list(ABSOLUTE_SCALE_COLUMNS)
        relative[:, columns] = flows[:, columns] - flows[:, columns].median(dim=0).values
    return relative


# Flow-channel candidates fixed on 2026-10-27 (configs/flow_variant_candidates.yaml).
FLOW_VARIANTS = {
    "all": _all,
    "activity_flags": _activity_flags,
    "composition": _composition,
    "window_relative": _window_relative,
}


def _number(value: str | None) -> float:
    cleaned = _clean(value)
    if cleaned is None:
        return math.nan
    lowered = cleaned.casefold()
    if lowered in {"true", "false"}:
        return 1.0 if lowered == "true" else 0.0
    try:
        return float(cleaned)
    except ValueError:
        return math.nan


@dataclass
class FlowRecords:
    sessions: list[str]
    timestamps: np.ndarray
    sources: list[str]
    destinations: list[str]
    protocols: list[str]
    services: list[str]
    values: np.ndarray  # columns follow NUMERIC_COLUMNS; nan is unavailable

    def __len__(self) -> int:
        return len(self.sessions)


def read_flow_records(path: str | Path) -> FlowRecords:
    """Read connection records with the session keys used by prepared graphs."""

    path = Path(path)
    session_namespace = f"file-0000:{path.name}"
    seen: set[bytes] = set()
    sessions: list[str] = []
    timestamps: list[int] = []
    sources: list[str] = []
    destinations: list[str] = []
    protocols: list[str] = []
    services: list[str] = []
    values: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        for row in reader:
            digest = hashlib.blake2b(
                "\x1f".join((row.get(field) or "") for field in fields).encode("utf-8"),
                digest_size=16,
            ).digest()
            if digest in seen:
                continue
            seen.add(digest)
            try:
                time_ns = _timestamp_ns(row.get("Timestamp"), row.get("Date"))
            except ValueError:
                continue
            source, destination = _clean(row.get("Scr_IP")), _clean(row.get("Des_IP"))
            if source is None or destination is None or source == destination:
                continue
            date = _clean(row.get("Date"))
            sessions.append(f"{session_namespace}:{date}" if date else session_namespace)
            timestamps.append(time_ns)
            sources.append(source)
            destinations.append(destination)
            protocols.append(_clean(row.get("Protocol")) or "<unavailable>")
            services.append(_clean(row.get("Service")) or "<unavailable>")
            values.append([_number(row.get(column)) for column in NUMERIC_COLUMNS])
    return FlowRecords(
        sessions,
        np.asarray(timestamps, dtype=np.int64),
        sources,
        destinations,
        protocols,
        services,
        np.asarray(values, dtype=np.float64).reshape(-1, len(NUMERIC_COLUMNS)),
    )


def window_key(sample: GraphSample) -> WindowKey:
    return sample.metadata["session_key"], int(sample.metadata["window_start_ns"]) // WINDOW_NS


def flow_table(
    records: FlowRecords, windows: Collection[WindowKey] | None = None
) -> dict[WindowKey, tuple[Tensor, list[tuple[str, str]]]]:
    """Flow vectors and their (source, destination) keys for each requested window."""

    window_ids = records.timestamps // WINDOW_NS
    grouped: dict[tuple[str, int, str, str], list[int]] = defaultdict(list)
    for index in range(len(records)):
        window = (records.sessions[index], int(window_ids[index]))
        if windows is not None and window not in windows:
            continue
        grouped[(*window, records.sources[index], records.destinations[index])].append(index)
    position = {name: column for column, name in enumerate(NUMERIC_COLUMNS)}
    vectors: dict[WindowKey, list[np.ndarray]] = defaultdict(list)
    keys: dict[WindowKey, list[tuple[str, str]]] = defaultdict(list)
    for (session, window, source, destination), indices in grouped.items():
        block = records.values[indices]

        def mean(name: str, block=block) -> float:
            column = block[:, position[name]]
            column = column[np.isfinite(column)]
            return float(column.mean()) if column.size else 0.0

        protocols = [records.protocols[i] for i in indices]
        services = [records.services[i] for i in indices]
        ports = block[:, position["Des_port"]]
        ports = ports[np.isfinite(ports)]
        vectors[(session, window)].append(
            np.asarray(
                [
                    math.log1p(len(indices)),
                    protocols.count("tcp") / len(indices),
                    protocols.count("udp") / len(indices),
                    protocols.count("icmp") / len(indices),
                    services.count("dns") / len(indices),
                    services.count("http") / len(indices),
                    services.count("other") / len(indices),
                    *(mean(flag) for flag in FLAGS),
                    math.log1p(max(mean("Duration"), 0.0)),
                    math.log1p(len(set(ports.tolist()))),
                    float(np.mean(ports < 1024)) if ports.size else 0.0,
                    math.log1p(max(mean("Scr_bytes"), 0.0)),
                    math.log1p(max(mean("Des_bytes"), 0.0)),
                ]
            )
        )
        keys[(session, window)].append((source, destination))
    return {key: (torch.from_numpy(np.stack(rows)), keys[key]) for key, rows in vectors.items()}


def flows_for(
    samples: Sequence[GraphSample],
    table: dict[WindowKey, tuple[Tensor, list[tuple[str, str]]]],
) -> tuple[list[Tensor], list[list[tuple[str, str]]]]:
    """Align flow context with graph windows; every window must have records."""

    missing = [window_key(sample) for sample in samples if window_key(sample) not in table]
    if missing:
        raise ValueError(f"{len(missing)} graph windows have no flow records, e.g. {missing[0]}")
    aligned = [table[window_key(sample)] for sample in samples]
    return [flows for flows, _ in aligned], [keys for _, keys in aligned]
