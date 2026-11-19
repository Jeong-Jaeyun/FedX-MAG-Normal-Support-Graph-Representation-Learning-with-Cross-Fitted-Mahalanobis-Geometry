"""Build session-isolated fixed-window graphs with label-independent boundaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from fedxmag.preprocessing.events import CanonicalEvent
from fedxmag.schema import (
    DEFAULT_FEATURE_GROUPS,
    RESOLUTION_AWARE_FEATURE_GROUPS,
    RESOLUTION_AWARE_V3A_FEATURE_GROUPS,
    FeatureGroups,
    GraphSample,
)

NANOSECONDS_PER_SECOND = 1_000_000_000


def _timing_features(timestamps_ns: Sequence[int]) -> tuple[list[float], list[bool]]:
    ordered = np.sort(np.asarray(timestamps_ns, dtype=np.int64))
    intervals_ms = np.diff(ordered).astype(np.float64) / 1_000_000.0
    values = [0.0, 0.0, 0.0, 0.0]
    available = [False, False, False, False]
    if intervals_ms.size >= 1:
        mean = float(intervals_ms.mean())
        values[0] = float(np.log1p(mean))
        values[3] = float(np.log1p(np.quantile(intervals_ms, 0.95)))
        available[0] = True
        available[3] = True
    if intervals_ms.size >= 2:
        deviation = float(intervals_ms.std(ddof=1))
        mean = float(intervals_ms.mean())
        values[1] = float(np.log1p(deviation))
        values[2] = (deviation - mean) / (deviation + mean + 1e-12)
        available[1] = True
        available[2] = True
    return values, available


def _resolution_aware_timing_features(
    timestamps_ns: Sequence[int], resolution_seconds: int
) -> tuple[list[float], list[bool]]:
    """Separate spacing between observable time bins from within-bin multiplicity.

    X-IIoTID timestamps are released at integer-second resolution. Repeated records
    in one second therefore do not establish a measured zero inter-arrival time.
    Spacing is computed only between distinct resolution bins, while collisions are
    retained as separate intensity statistics.
    """

    if resolution_seconds < 1:
        raise ValueError("timestamp resolution must be a positive number of seconds")
    resolution_ns = resolution_seconds * NANOSECONDS_PER_SECOND
    ordered = np.sort(np.asarray(timestamps_ns, dtype=np.int64))
    bins, multiplicities = np.unique(ordered // resolution_ns, return_counts=True)
    gaps_seconds = np.diff(bins).astype(np.float64) * float(resolution_seconds)
    values = [0.0, 0.0, 0.0, 0.0, 0.0]
    available = [False, False, False, False, False]

    if gaps_seconds.size >= 1:
        values[0] = float(np.log1p(np.median(gaps_seconds)))
        available[0] = True
    if gaps_seconds.size >= 2:
        values[1] = float(
            np.log1p(
                np.quantile(gaps_seconds, 0.75) - np.quantile(gaps_seconds, 0.25)
            )
        )
        values[2] = float(np.log1p(np.quantile(gaps_seconds, 0.95)))
        available[1] = True
        available[2] = True
    if ordered.size >= 2:
        values[3] = 1.0 - float(bins.size) / float(ordered.size)
        values[4] = float(np.log1p(np.quantile(multiplicities, 0.95)))
        available[3] = True
        available[4] = True
    return values, available


def _timing_features_for_profile(
    timestamps_ns: Sequence[int], timing_profile: str, resolution_seconds: int
) -> tuple[list[float], list[bool]]:
    if timing_profile == "legacy_record_iat_v2":
        return _timing_features(timestamps_ns)
    if timing_profile == "distinct_second_multiplicity_v3":
        return _resolution_aware_timing_features(timestamps_ns, resolution_seconds)
    if timing_profile == "distinct_second_collision_v3":
        values, available = _resolution_aware_timing_features(
            timestamps_ns, resolution_seconds
        )
        return values[:4], available[:4]
    raise ValueError(f"Unknown timing feature profile: {timing_profile}")


def _distribution_features(
    sizes: Sequence[float | None],
) -> tuple[list[float], list[bool]]:
    observed = [value for value in sizes if value is not None]
    observations = np.asarray(observed, dtype=np.float64)
    values = [0.0, 0.0, 0.0]
    available = [False, False, False]
    if observations.size >= 1:
        values[2] = float(np.log1p(np.quantile(observations, 0.90)))
        available[2] = True
    if observations.size >= 2:
        values[0] = float(np.log1p(observations.std(ddof=1)))
        values[1] = float(
            np.log1p(np.quantile(observations, 0.75) - np.quantile(observations, 0.25))
        )
        available[0] = True
        available[1] = True
    return values, available


def _build_one(
    session_key: str,
    window_id: int,
    events: Sequence[CanonicalEvent],
    window_ns: int,
    groups: FeatureGroups,
    timing_profile: str,
    timestamp_resolution_seconds: int,
) -> GraphSample:
    node_keys = sorted({event.src_key for event in events} | {event.dst_key for event in events})
    node_to_index = {key: index for index, key in enumerate(node_keys)}
    edge_keys = sorted({(event.src_key, event.dst_key) for event in events})

    node_out: dict[str, list[CanonicalEvent]] = defaultdict(list)
    node_in: dict[str, list[CanonicalEvent]] = defaultdict(list)
    by_edge: dict[tuple[str, str], list[CanonicalEvent]] = defaultdict(list)
    for event in events:
        node_out[event.src_key].append(event)
        node_in[event.dst_key].append(event)
        by_edge[(event.src_key, event.dst_key)].append(event)

    node_values: list[list[float]] = []
    node_available: list[list[bool]] = []
    for key in node_keys:
        outgoing = node_out[key]
        incoming = node_in[key]
        out_timing, out_timing_available = _timing_features_for_profile(
            [event.timestamp_ns for event in outgoing if event.timing_observed],
            timing_profile,
            timestamp_resolution_seconds,
        )
        in_timing, in_timing_available = _timing_features_for_profile(
            [event.timestamp_ns for event in incoming if event.timing_observed],
            timing_profile,
            timestamp_resolution_seconds,
        )
        out_distribution, out_distribution_available = _distribution_features(
            [event.directional_mean_packet_size for event in outgoing]
        )
        in_distribution, in_distribution_available = _distribution_features(
            [event.directional_mean_packet_size for event in incoming]
        )
        node_values.append(out_timing + in_timing + out_distribution + in_distribution)
        node_available.append(
            out_timing_available
            + in_timing_available
            + out_distribution_available
            + in_distribution_available
        )

    edge_values: list[list[float]] = []
    edge_available: list[list[bool]] = []
    edge_index: list[list[int]] = [[], []]
    for source, destination in edge_keys:
        observations = by_edge[(source, destination)]
        timing, timing_available = _timing_features_for_profile(
            [event.timestamp_ns for event in observations if event.timing_observed],
            timing_profile,
            timestamp_resolution_seconds,
        )
        distribution, distribution_available = _distribution_features(
            [event.directional_mean_packet_size for event in observations]
        )
        edge_values.append(timing + distribution)
        edge_available.append(timing_available + distribution_available)
        edge_index[0].append(node_to_index[source])
        edge_index[1].append(node_to_index[destination])

    protocols = Counter(event.protocol or "<unavailable>" for event in events)
    attack_categories = sorted(
        {event.attack_category for event in events if event.attack_category is not None}
    )
    attack_subcategories = sorted(
        {
            event.attack_subcategory
            for event in events
            if event.attack_subcategory is not None
        }
    )
    sample = GraphSample(
        node_features=torch.tensor(node_values, dtype=torch.float32),
        edge_features=torch.tensor(edge_values, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        node_available=torch.tensor(node_available, dtype=torch.bool),
        edge_available=torch.tensor(edge_available, dtype=torch.bool),
        label=int(any(event.is_attack for event in events)),
        metadata={
            "session_key": session_key,
            "window_start_ns": window_id * window_ns,
            "window_end_ns": (window_id + 1) * window_ns,
            "endpoint_keys": node_keys,
            "edge_keys": [list(edge) for edge in edge_keys],
            "protocol_counts": dict(protocols),
            "attack_categories": attack_categories,
            "attack_subcategories": attack_subcategories,
            "event_count": len(events),
            "timing_observed_event_count": sum(
                event.timing_observed for event in events
            ),
            "synthetic_reverse_event_count": sum(
                not event.timing_observed for event in events
            ),
            "size_available_event_count": sum(
                event.directional_mean_packet_size is not None for event in events
            ),
            "timing_feature_profile": timing_profile,
            "timestamp_resolution_seconds": timestamp_resolution_seconds,
        },
    )
    sample.validate(groups)
    return sample


def build_static_graphs(
    events: Sequence[CanonicalEvent],
    window_seconds: int = 60,
    *,
    groups: FeatureGroups = DEFAULT_FEATURE_GROUPS,
    timing_profile: str = "legacy_record_iat_v2",
    timestamp_resolution_seconds: int = 1,
) -> list[GraphSample]:
    if window_seconds < 1:
        raise ValueError("window_seconds must be positive")
    if timing_profile == "distinct_second_multiplicity_v3":
        if groups != RESOLUTION_AWARE_V3A_FEATURE_GROUPS:
            raise ValueError("The v3a timing profile requires its five-feature schema")
    elif timing_profile == "distinct_second_collision_v3":
        if groups != RESOLUTION_AWARE_FEATURE_GROUPS:
            raise ValueError("The v3 timing profile requires its resolution-aware schema")
    elif groups != DEFAULT_FEATURE_GROUPS:
        raise ValueError("The legacy timing profile requires the legacy schema")
    window_ns = window_seconds * NANOSECONDS_PER_SECOND
    windows: dict[tuple[str, int], list[CanonicalEvent]] = defaultdict(list)
    for event in events:
        if event.src_key != event.dst_key:
            key = (event.session_key, event.timestamp_ns // window_ns)
            windows[key].append(event)
    return [
        _build_one(
            session,
            window_id,
            windows[(session, window_id)],
            window_ns,
            groups,
            timing_profile,
            timestamp_resolution_seconds,
        )
        for session, window_id in sorted(windows)
    ]


def _split_ordered_items[T](
    ordered_items: Sequence[T], fractions: Sequence[float]
) -> list[list[T]]:
    if len(ordered_items) < len(fractions):
        raise ValueError(
            f"Need at least {len(fractions)} units for {len(fractions)} nonempty splits"
        )
    cumulative = np.cumsum(fractions)[:-1]
    boundaries: list[int] = []
    for index, point in enumerate(cumulative):
        minimum = index + 1
        maximum = len(ordered_items) - (len(fractions) - index - 1)
        boundaries.append(min(max(int(round(point * len(ordered_items))), minimum), maximum))
    result: list[list[T]] = []
    start = 0
    for end in boundaries:
        result.append(list(ordered_items[start:end]))
        start = end
    result.append(list(ordered_items[start:]))
    return result


def chronological_window_aligned_split(
    events: Sequence[CanonicalEvent],
    fractions: Mapping[str, float],
    window_seconds: int = 60,
    session_split_strategy: str = "exclusive_if_possible",
) -> dict[str, list[CanonicalEvent]]:
    """Assign complete session-windows without consulting labels.

    Labels never participate in split construction. No `(session, window)` pair
    can occur in more than one output partition.
    """

    if not events:
        raise ValueError("Cannot split an empty event collection")
    if not fractions or any(value <= 0 for value in fractions.values()):
        raise ValueError("Every split fraction must be positive")
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Split fractions must sum to one")
    if window_seconds < 1:
        raise ValueError("window_seconds must be positive")
    if session_split_strategy not in {
        "exclusive_if_possible",
        "within_session_chronological",
    }:
        raise ValueError(f"Unknown session split strategy: {session_split_strategy}")

    split_names = list(fractions)
    split_fractions = [fractions[name] for name in split_names]
    window_ns = window_seconds * NANOSECONDS_PER_SECOND
    sessions: dict[str, list[CanonicalEvent]] = defaultdict(list)
    for event in events:
        sessions[event.session_key].append(event)
    result = {name: [] for name in split_names}

    if session_split_strategy == "within_session_chronological":
        for session_events in sessions.values():
            window_ids = sorted(
                {event.timestamp_ns // window_ns for event in session_events}
            )
            window_groups = _split_ordered_items(window_ids, split_fractions)
            assignment = {
                int(window_id): name
                for name, group in zip(split_names, window_groups, strict=True)
                for window_id in group
            }
            for event in session_events:
                result[assignment[event.timestamp_ns // window_ns]].append(event)
        return result

    if len(sessions) >= len(split_names):
        ordered_sessions = sorted(
            sessions, key=lambda key: (min(event.timestamp_ns for event in sessions[key]), key)
        )
        session_groups = _split_ordered_items(ordered_sessions, split_fractions)
        for name, group in zip(split_names, session_groups, strict=True):
            for session in group:
                result[name].extend(sessions[session])
        return result

    window_keys = sorted(
        {
            (event.session_key, event.timestamp_ns // window_ns)
            for event in events
        },
        key=lambda item: (item[1], item[0]),
    )
    window_groups = _split_ordered_items(window_keys, split_fractions)
    assignment = {
        window_key: name
        for name, group in zip(split_names, window_groups, strict=True)
        for window_key in group
    }
    for event in events:
        key = (event.session_key, event.timestamp_ns // window_ns)
        result[assignment[key]].append(event)
    return result
