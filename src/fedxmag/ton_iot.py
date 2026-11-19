"""TON_IoT processed network tables as v3b graph windows and flow context.

Rows follow the ingestion rules of :mod:`fedxmag.preprocessing.events`: the label is
validated, exact duplicate rows (over all fields, across tables) are dropped, then
rows with a bad timestamp, a missing endpoint, or a self-relation. Each kept row
becomes the same forward event and distribution-only synthetic reverse event as an
X-IIoTID row, so the frozen graph builder computes the v3b features unchanged.

A capture is a testbed calendar day in AEST (UTC+10, fixed offset), chosen from the
label-free layout census before any label was counted. TON_IoT has no TCP flag
columns, so the five X-IIoTID flag rates of the flow context are derived from the
Zeek connection state (TCP only; other protocols carry no flags), and a Zeek
service of ``-`` (none identified) counts as ``other``. ``label`` and ``type`` are
kept only to label windows and build label-0 background views for evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import math
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from torch import Tensor

from fedxmag.flow_context import NUMERIC_COLUMNS, WINDOW_NS, FlowRecords, _number, flow_table
from fedxmag.preprocessing.events import CanonicalEvent, _clean, _parse_label, _timestamp_ns
from fedxmag.preprocessing.graph_builder import build_static_graphs
from fedxmag.schema import FeatureGroups, GraphSample

CAPTURE_OFFSET_SECONDS = 10 * 3600
SESSION_PREFIX = "ton_iot"
# Flag column -> Zeek TCP connection states in which that packet kind was seen.
# is_syn_only: originator SYN; Is_SYN_ACK: responder SYN-ACK; is_pure_ack: the
# handshake-completing ACK of an established connection; FIN or RST: a FIN or RST.
FLAG_STATES = {
    "is_syn_only": frozenset({"S0", "S1", "SF", "REJ", "S2", "S3", "RSTO", "RSTR", "RSTOS0", "SH"}),
    "FIN or RST": frozenset(
        {"SF", "S2", "S3", "RSTO", "RSTR", "RSTOS0", "RSTRH", "SH", "SHR", "REJ"}
    ),
    "is_pure_ack": frozenset({"S1", "SF", "S2", "S3", "RSTO", "RSTR"}),
    "Is_SYN_ACK": frozenset({"S1", "SF", "S2", "S3", "RSTO", "RSTR", "RSTRH", "SHR"}),
}
UNIDENTIFIED_SERVICE = "other"
COUNTERS = ("src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "duration", "dst_port")


def capture_of(timestamp_ns: int) -> str:
    seconds = timestamp_ns // 1_000_000_000 + CAPTURE_OFFSET_SECONDS
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%d")


def session_key(capture: str) -> str:
    return f"{SESSION_PREFIX}:{capture}"


class _Codes:
    def __init__(self) -> None:
        self.values: list[str | None] = []
        self.index: dict[str | None, int] = {}

    def __call__(self, value: str | None) -> int:
        code = self.index.get(value)
        if code is None:
            code = self.index[value] = len(self.values)
            self.values.append(value)
        return code


@dataclass
class TonRecords:
    """Kept rows in table order; codes index the vocabularies."""

    timestamps: np.ndarray  # int64 ns
    sources: np.ndarray
    destinations: np.ndarray
    endpoints: list[str]
    protocols: np.ndarray
    protocol_names: list[str | None]
    services: np.ndarray
    service_names: list[str | None]
    states: np.ndarray
    state_names: list[str | None]
    counters: np.ndarray  # columns follow COUNTERS; nan is missing or invalid
    attack: np.ndarray  # bool
    types: np.ndarray
    type_names: list[str | None]
    report: dict[str, int]

    def __len__(self) -> int:
        return len(self.timestamps)

    def windows(self) -> dict[int, np.ndarray]:
        """Row indices of each 60 s window (UTC-minute aligned), in table order."""

        minutes = self.timestamps // WINDOW_NS
        order = np.argsort(minutes, kind="stable")
        ordered = minutes[order]
        starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        ends = np.r_[starts[1:], len(order)]
        return {int(ordered[a]): order[a:b] for a, b in zip(starts, ends, strict=True)}

    def events(self, rows: Sequence[int]) -> list[CanonicalEvent]:
        """Forward and synthetic reverse events exactly as the X-IIoTID reader makes them."""

        events = []
        for row in rows:
            time_ns = int(self.timestamps[row])
            source, destination = self.endpoints[self.sources[row]], self.endpoints[
                self.destinations[row]
            ]
            src_bytes, dst_bytes, src_pkts, dst_pkts = (
                _nonnegative(value) for value in self.counters[row, :4]
            )
            attack = bool(self.attack[row])
            category = self.type_names[self.types[row]] if attack else None
            protocol = self.protocol_names[self.protocols[row]]
            session = session_key(capture_of(time_ns))
            events.append(
                CanonicalEvent(
                    time_ns, source, destination, _mean_size(src_bytes, src_pkts), protocol,
                    attack, session, category,
                )
            )
            if (dst_pkts is not None and dst_pkts > 0) or (dst_bytes is not None and dst_bytes > 0):
                events.append(
                    CanonicalEvent(
                        time_ns, destination, source, _mean_size(dst_bytes, dst_pkts), protocol,
                        attack, session, category, timing_observed=False,
                    )
                )
        return events

    def flow_records(self, rows: Sequence[int]) -> FlowRecords:
        values = np.full((len(rows), len(NUMERIC_COLUMNS)), np.nan)
        position = {name: column for column, name in enumerate(NUMERIC_COLUMNS)}
        for line, row in enumerate(rows):
            counters = dict(zip(COUNTERS, self.counters[row].tolist(), strict=True))
            state = self.state_names[self.states[row]]
            tcp = self.protocol_names[self.protocols[row]] == "tcp"
            for name, states in FLAG_STATES.items():
                if not tcp:
                    values[line, position[name]] = 0.0
                elif state is not None:
                    values[line, position[name]] = float(state in states)
            sizes = [counters["src_bytes"], counters["dst_bytes"]]
            values[line, position["is_with_payload"]] = (
                math.nan if all(math.isnan(v) for v in sizes)
                else float(any(v > 0 for v in sizes if not math.isnan(v)))
            )
            values[line, position["Duration"]] = counters["duration"]
            values[line, position["Des_port"]] = counters["dst_port"]
            values[line, position["Scr_bytes"]] = counters["src_bytes"]
            values[line, position["Des_bytes"]] = counters["dst_bytes"]
        services = []
        for row in rows:
            service = self.service_names[self.services[row]]
            services.append(UNIDENTIFIED_SERVICE if service is None else service)
        return FlowRecords(
            [session_key(capture_of(int(self.timestamps[row]))) for row in rows],
            self.timestamps[list(rows)].astype(np.int64),
            [self.endpoints[self.sources[row]] for row in rows],
            [self.endpoints[self.destinations[row]] for row in rows],
            [self.protocol_names[self.protocols[row]] or "<unavailable>" for row in rows],
            services,
            values,
        )

    def view(
        self,
        rows: Sequence[int],
        groups: FeatureGroups,
        *,
        timing_profile: str,
        timestamp_resolution_seconds: int,
    ) -> tuple[GraphSample, Tensor, list[tuple[str, str]]]:
        """One window's raw graph, flow-context matrix, and flow keys from these rows."""

        graphs = build_static_graphs(
            self.events(rows),
            WINDOW_NS // 1_000_000_000,
            groups=groups,
            timing_profile=timing_profile,
            timestamp_resolution_seconds=timestamp_resolution_seconds,
        )
        table = flow_table(self.flow_records(rows))
        if len(graphs) != 1 or len(table) != 1:
            raise ValueError("A view must cover exactly one capture window")
        (flows, keys), = table.values()
        return graphs[0], flows, keys


def _nonnegative(value: float) -> float | None:
    return None if not math.isfinite(value) or value < 0 else float(value)


def _mean_size(byte_count: float | None, packet_count: float | None) -> float | None:
    if byte_count is None or packet_count is None or packet_count <= 0:
        return None
    return byte_count / packet_count


def read_ton_records(paths: Sequence[str | Path]) -> TonRecords:
    """Read processed network tables in the given order with one shared duplicate set."""

    seen: set[bytes] = set()
    report = dict.fromkeys(
        ("rows_total", "exact_duplicate", "bad_timestamp", "missing_endpoint", "self_relation",
         "kept"),
        0,
    )
    timestamps, sources, destinations = array("q"), array("i"), array("i")
    protocols, services, states, types = array("h"), array("h"), array("h"), array("h")
    counters, attack = array("d"), array("b")
    endpoints, protocol_codes, service_codes = _Codes(), _Codes(), _Codes()
    state_codes, type_codes = _Codes(), _Codes()
    for path in paths:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            fields = next(reader)
            width = len(fields)
            column = {name: fields.index(name) for name in (
                "ts", "src_ip", "dst_ip", "proto", "service", "conn_state", "label", "type",
                *COUNTERS,
            )}
            for line, row in enumerate(reader, start=2):
                if not row:
                    continue
                report["rows_total"] += 1
                values = (row + [""] * width)[:width]
                try:
                    is_attack = _parse_label(values[column["label"]])
                except ValueError as error:
                    raise ValueError(f"Invalid label at {path}:{line}: {error}") from error
                digest = hashlib.blake2b(
                    "\x1f".join(values).encode("utf-8"), digest_size=16
                ).digest()
                if digest in seen:
                    report["exact_duplicate"] += 1
                    continue
                seen.add(digest)
                try:
                    time_ns = _timestamp_ns(values[column["ts"]], None)
                except ValueError:
                    report["bad_timestamp"] += 1
                    continue
                source = _clean(values[column["src_ip"]])
                destination = _clean(values[column["dst_ip"]])
                if source is None or destination is None:
                    report["missing_endpoint"] += 1
                    continue
                if source == destination:
                    report["self_relation"] += 1
                    continue
                report["kept"] += 1
                timestamps.append(time_ns)
                sources.append(endpoints(source))
                destinations.append(endpoints(destination))
                protocols.append(protocol_codes(_clean(values[column["proto"]])))
                services.append(service_codes(_clean(values[column["service"]])))
                states.append(state_codes(_clean(values[column["conn_state"]])))
                types.append(type_codes(_clean(values[column["type"]])))
                attack.append(is_attack)
                counters.extend(_number(values[column[name]]) for name in COUNTERS)
    return TonRecords(
        np.frombuffer(timestamps, dtype=np.int64).copy(),
        np.frombuffer(sources, dtype=np.int32).copy(),
        np.frombuffer(destinations, dtype=np.int32).copy(),
        [str(value) for value in endpoints.values],
        np.frombuffer(protocols, dtype=np.int16).copy(),
        protocol_codes.values,
        np.frombuffer(services, dtype=np.int16).copy(),
        service_codes.values,
        np.frombuffer(states, dtype=np.int16).copy(),
        state_codes.values,
        np.frombuffer(counters, dtype=np.float64).reshape(-1, len(COUNTERS)).copy(),
        np.frombuffer(attack, dtype=np.int8).astype(bool),
        np.frombuffer(types, dtype=np.int16).copy(),
        type_codes.values,
        report,
    )
