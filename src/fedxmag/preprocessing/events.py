"""Convert X-IIoTID rows into validated direction-specific observations."""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

MISSING_SENTINELS = frozenset({"", "-", "?", "nan", "none", "null", "n/a", "na"})
NORMAL_LABELS = frozenset({"0", "0.0", "normal", "benign", "false"})
ATTACK_LABELS = frozenset({"1", "1.0", "attack", "malicious", "true"})


@dataclass(frozen=True, slots=True)
class CanonicalEvent:
    """One observed directed communication event.

    `directional_mean_packet_size=None` means that the event and its timestamp
    are observed while the directional byte/packet counters are unavailable.
    `timing_observed=False` marks a reverse-direction size observation synthesized
    from bidirectional counters in one connection row. Such an observation may
    contribute to distribution features, but its row timestamp is not an
    independently observed reverse-direction event time.
    """

    timestamp_ns: int
    src_key: str
    dst_key: str
    directional_mean_packet_size: float | None
    protocol: str | None
    is_attack: bool
    session_key: str
    attack_category: str | None = None
    attack_subcategory: str | None = None
    timing_observed: bool = True

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be nonnegative")
        if not self.src_key or not self.dst_key:
            raise ValueError("Source and destination keys must be nonempty")
        if not self.session_key:
            raise ValueError("session_key must be nonempty")
        size = self.directional_mean_packet_size
        if size is not None and (not math.isfinite(size) or size < 0):
            raise ValueError("directional_mean_packet_size must be finite and nonnegative")


@dataclass
class IngestionReport:
    files: int = 0
    rows_total: int = 0
    rows_skipped_bad_timestamp: int = 0
    rows_skipped_missing_endpoint: int = 0
    rows_skipped_self_relation: int = 0
    exact_duplicate_rows_detected: int = 0
    rows_skipped_exact_duplicate: int = 0
    counter_values_missing: int = 0
    counter_values_invalid: int = 0
    forward_events: int = 0
    reverse_events: int = 0
    timing_observed_events: int = 0
    timing_observed_subsecond_events: int = 0
    timing_observed_same_timestamp_edge_collisions: int = 0
    timing_observed_same_timestamp_node_out_collisions: int = 0
    timing_observed_same_timestamp_node_in_collisions: int = 0
    synthetic_reverse_events_excluded_from_timing: int = 0
    reverse_direction_unknown: int = 0
    size_unavailable_events: int = 0
    protocol_unavailable_rows: int = 0
    repeated_event_signatures: int = 0
    raw_label_values: dict[str, int] = field(default_factory=dict)
    label_values: dict[str, int] = field(default_factory=dict)
    counter_missing_by_field: dict[str, int] = field(default_factory=dict)
    counter_invalid_by_field: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EventReadResult:
    events: list[CanonicalEvent]
    report: IngestionReport


@dataclass(frozen=True)
class XIIoTIDColumns:
    """Column aliases used by released and paper-described X-IIoTID tables."""

    timestamp: tuple[str, ...] = ("Timestamp", "timestamp", "Ts", "ts")
    date: tuple[str, ...] = ("Date", "date")
    source: tuple[str, ...] = (
        "Scr_IP", "Scr IP", "Src_IP", "Src IP", "src_ip", "Source IP", "id.orig_h"
    )
    destination: tuple[str, ...] = (
        "Des_IP", "Des IP", "Dst_IP", "Dst IP", "dst_ip", "Destination IP", "id.resp_h"
    )
    source_bytes: tuple[str, ...] = (
        "Scr_bytes", "Scr bytes", "Src_bytes", "Src bytes", "src_bytes", "orig_bytes"
    )
    destination_bytes: tuple[str, ...] = (
        "Des_bytes", "Des bytes", "Dst_bytes", "Dst bytes", "dst_bytes", "resp_bytes"
    )
    source_packets: tuple[str, ...] = (
        "Scr_pkts", "Scr pkts", "Src_pkts", "Src pkts", "src_pkts", "orig_pkts"
    )
    destination_packets: tuple[str, ...] = (
        "Des_pkts", "Des pkts", "Dst_pkts", "Dst pkts", "dst_pkts", "resp_pkts"
    )
    protocol: tuple[str, ...] = ("Protocol", "protocol", "proto")
    # Released X-IIoTID uses class3=binary, class2=category, class1=subtype.
    # Generic aliases remain as fallbacks for converted tables.
    label: tuple[str, ...] = ("class3", "Class3", "label", "Label", "class1", "Class1")
    attack_category: tuple[str, ...] = (
        "class2", "Class2", "attack_category", "Attack_type"
    )
    attack_subcategory: tuple[str, ...] = (
        "class1", "Class1", "attack_subcategory"
    )
    session: tuple[str, ...] = ("session_id", "capture_id", "file_id", "Date", "date")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return None if cleaned.casefold() in MISSING_SENTINELS else cleaned


def _resolve(fieldnames: Sequence[str], aliases: Iterable[str], required: bool) -> str | None:
    normalized = {name.strip().casefold(): name for name in fieldnames}
    for alias in aliases:
        match = normalized.get(alias.strip().casefold())
        if match is not None:
            return match
    if required:
        raise ValueError(f"None of the required column aliases were found: {tuple(aliases)}")
    return None


def _numeric_timestamp_ns(value: float) -> int:
    magnitude = abs(value)
    if magnitude >= 1e17:
        return int(value)
    if magnitude >= 1e14:
        return int(value * 1_000)
    if magnitude >= 1e11:
        return int(value * 1_000_000)
    return int(value * 1_000_000_000)


def _parse_date(value: str) -> datetime:
    for pattern in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, pattern).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date value: {value!r}")


def _parse_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        pass
    for pattern in (
        "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            continue
    return None


def _timestamp_ns(value: str | None, date_value: str | None) -> int:
    raw = _clean(value)
    date_raw = _clean(date_value)
    if raw is None:
        raise ValueError("Missing timestamp")
    try:
        numeric = float(raw)
    except ValueError:
        numeric = None
    if numeric is not None:
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError(f"Invalid numeric timestamp: {raw!r}")
        if date_raw is not None and numeric < 86_400:
            parsed = _parse_date(date_raw) + timedelta(seconds=numeric)
            return int(parsed.timestamp() * 1_000_000_000)
        return _numeric_timestamp_ns(numeric)

    parsed = _parse_datetime(raw)
    if parsed is None and date_raw is not None:
        parsed = _parse_datetime(f"{date_raw} {raw}")
    if parsed is None:
        raise ValueError(f"Unsupported timestamp value: {raw!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1_000_000_000)


def _parse_label(value: str | None) -> bool:
    cleaned = _clean(value)
    if cleaned is None:
        raise ValueError("Missing binary label")
    normalized = cleaned.casefold()
    if normalized in NORMAL_LABELS:
        return False
    if normalized in ATTACK_LABELS:
        return True
    raise ValueError(f"Unknown binary label: {cleaned!r}")


def _optional_nonnegative(
    value: str | None, report: IngestionReport, field_name: str
) -> float | None:
    cleaned = _clean(value)
    if cleaned is None:
        report.counter_values_missing += 1
        report.counter_missing_by_field[field_name] = (
            report.counter_missing_by_field.get(field_name, 0) + 1
        )
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        report.counter_values_invalid += 1
        report.counter_invalid_by_field[field_name] = (
            report.counter_invalid_by_field.get(field_name, 0) + 1
        )
        return None
    if not math.isfinite(parsed) or parsed < 0:
        report.counter_values_invalid += 1
        report.counter_invalid_by_field[field_name] = (
            report.counter_invalid_by_field.get(field_name, 0) + 1
        )
        return None
    return parsed


def _mean_size(byte_count: float | None, packet_count: float | None) -> float | None:
    if byte_count is None or packet_count is None or packet_count <= 0:
        return None
    return byte_count / packet_count


def _read_one(
    path: Path,
    aliases: XIIoTIDColumns,
    report: IngestionReport,
    file_namespace: str,
    seen_row_digests: set[bytes],
    deduplicate_exact_rows: bool,
) -> Iterator[CanonicalEvent]:
    report.files += 1
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fields = reader.fieldnames
        timestamp = _resolve(fields, aliases.timestamp, True)
        date = _resolve(fields, aliases.date, False)
        source = _resolve(fields, aliases.source, True)
        destination = _resolve(fields, aliases.destination, True)
        source_bytes = _resolve(fields, aliases.source_bytes, False)
        destination_bytes = _resolve(fields, aliases.destination_bytes, False)
        source_packets = _resolve(fields, aliases.source_packets, False)
        destination_packets = _resolve(fields, aliases.destination_packets, False)
        protocol = _resolve(fields, aliases.protocol, False)
        label = _resolve(fields, aliases.label, True)
        category = _resolve(fields, aliases.attack_category, False)
        subcategory = _resolve(fields, aliases.attack_subcategory, False)
        if subcategory == label:
            subcategory = None
        session = _resolve(fields, aliases.session, False)
        if timestamp is None or source is None or destination is None or label is None:
            raise AssertionError("Required columns were resolved to None")

        for row_number, row in enumerate(reader, start=2):
            report.rows_total += 1
            raw_label = _clean(row.get(label))
            label_key = raw_label.casefold() if raw_label is not None else "<missing>"
            report.raw_label_values[label_key] = (
                report.raw_label_values.get(label_key, 0) + 1
            )
            try:
                attack = _parse_label(raw_label)
            except ValueError as error:
                raise ValueError(f"Invalid label at {path}:{row_number}: {error}") from error

            digest = hashlib.blake2b(
                "\x1f".join((row.get(field) or "") for field in fields).encode("utf-8"),
                digest_size=16,
            ).digest()
            if digest in seen_row_digests:
                report.exact_duplicate_rows_detected += 1
                if deduplicate_exact_rows:
                    report.rows_skipped_exact_duplicate += 1
                    continue
            else:
                seen_row_digests.add(digest)

            try:
                time_ns = _timestamp_ns(row.get(timestamp), row.get(date) if date else None)
            except ValueError:
                report.rows_skipped_bad_timestamp += 1
                continue
            src = _clean(row.get(source))
            dst = _clean(row.get(destination))
            if src is None or dst is None:
                report.rows_skipped_missing_endpoint += 1
                continue
            if src == dst:
                report.rows_skipped_self_relation += 1
                continue
            report.label_values[label_key] = report.label_values.get(label_key, 0) + 1

            source_session = _clean(row.get(session)) if session else None
            session_key = (
                f"{file_namespace}:{source_session}"
                if source_session is not None
                else file_namespace
            )
            proto = _clean(row.get(protocol)) if protocol else None
            if proto is None:
                report.protocol_unavailable_rows += 1
            attack_category = _clean(row.get(category)) if attack and category else None
            attack_subcategory = (
                _clean(row.get(subcategory)) if attack and subcategory else None
            )

            src_bytes = _optional_nonnegative(
                row.get(source_bytes) if source_bytes else None, report, "source_bytes"
            )
            dst_bytes = _optional_nonnegative(
                row.get(destination_bytes) if destination_bytes else None,
                report,
                "destination_bytes",
            )
            src_packets = _optional_nonnegative(
                row.get(source_packets) if source_packets else None,
                report,
                "source_packets",
            )
            dst_packets = _optional_nonnegative(
                row.get(destination_packets) if destination_packets else None,
                report,
                "destination_packets",
            )

            forward_size = _mean_size(src_bytes, src_packets)
            if forward_size is None:
                report.size_unavailable_events += 1
            report.forward_events += 1
            report.timing_observed_events += 1
            yield CanonicalEvent(
                time_ns,
                src,
                dst,
                forward_size,
                proto,
                attack,
                session_key,
                attack_category,
                attack_subcategory,
            )

            reverse_observed = (dst_packets is not None and dst_packets > 0) or (
                dst_bytes is not None and dst_bytes > 0
            )
            if reverse_observed:
                reverse_size = _mean_size(dst_bytes, dst_packets)
                if reverse_size is None:
                    report.size_unavailable_events += 1
                report.reverse_events += 1
                yield CanonicalEvent(
                    time_ns,
                    dst,
                    src,
                    reverse_size,
                    proto,
                    attack,
                    session_key,
                    attack_category,
                    attack_subcategory,
                    timing_observed=False,
                )
                report.synthetic_reverse_events_excluded_from_timing += 1
            elif dst_packets is None or dst_bytes is None:
                report.reverse_direction_unknown += 1


def read_xiiotid_events(
    paths: str | Path | Sequence[str | Path],
    columns: XIIoTIDColumns | None = None,
    *,
    deduplicate_exact_rows: bool = True,
) -> EventReadResult:
    """Read X-IIoTID CSV files with explicit missingness and label validation."""

    selected_paths = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not selected_paths:
        raise ValueError("At least one X-IIoTID CSV path is required")
    resolved_paths = [Path(path).resolve() for path in selected_paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("The same X-IIoTID input path was supplied more than once")
    aliases = columns or XIIoTIDColumns()
    report = IngestionReport()
    seen_row_digests: set[bytes] = set()
    events: list[CanonicalEvent] = []
    for index, path in enumerate(resolved_paths):
        namespace = f"file-{index:04d}:{path.name}"
        events.extend(
            _read_one(
                path,
                aliases,
                report,
                namespace,
                seen_row_digests,
                deduplicate_exact_rows,
            )
        )
    events.sort(
        key=lambda event: (
            event.session_key,
            event.timestamp_ns,
            event.src_key,
            event.dst_key,
        )
    )
    signatures = [
        (event.session_key, event.timestamp_ns, event.src_key, event.dst_key,
         event.directional_mean_packet_size, event.timing_observed)
        for event in events
    ]
    report.repeated_event_signatures = len(signatures) - len(set(signatures))
    timing_events = [event for event in events if event.timing_observed]
    report.timing_observed_subsecond_events = sum(
        event.timestamp_ns % 1_000_000_000 != 0 for event in timing_events
    )
    for attribute, keys in (
        (
            "timing_observed_same_timestamp_edge_collisions",
            {
                (event.session_key, event.timestamp_ns, event.src_key, event.dst_key)
                for event in timing_events
            },
        ),
        (
            "timing_observed_same_timestamp_node_out_collisions",
            {
                (event.session_key, event.timestamp_ns, event.src_key)
                for event in timing_events
            },
        ),
        (
            "timing_observed_same_timestamp_node_in_collisions",
            {
                (event.session_key, event.timestamp_ns, event.dst_key)
                for event in timing_events
            },
        ),
    ):
        setattr(report, attribute, len(timing_events) - len(keys))
    return EventReadResult(events=events, report=report)
