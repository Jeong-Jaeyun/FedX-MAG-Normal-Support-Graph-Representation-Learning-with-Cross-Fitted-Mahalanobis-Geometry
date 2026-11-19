"""Anonymous, strictly past connection behavior from unlabelled event tables.

Addresses index transient counters; their values never enter model features.
Every connection in a completed-second group queries history before any member
of that group is inserted. Histories include all traffic, without label filters.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .experiments import sha256, write_json
from .record_data import NUMERIC, XIIOTID_COLUMNS
from .record_experiment import corruption_groups

SCOPES = ("outgoing", "incoming", "pair_service")
STATISTICS = ("log_count", "log_peers", "log_ports", "failed_fraction", "log_bytes", "log_packets")
HORIZONS = (60, 300)
CONTEXT_FEATURES = tuple(
    f"context_{scope}_{seconds}s_{statistic}"
    for seconds in HORIZONS
    for scope in SCOPES
    for statistic in STATISTICS
)
FAILED_STATES = frozenset(("S0", "REJ", "RSTO", "RSTR", "RSTOS0", "RSTRH"))


class RollingBehavior:
    def __init__(self, horizon: int):
        self.horizon = horizon
        self.events = deque()
        self.stats = [{} for _ in SCOPES]

    @staticmethod
    def keys(event: tuple) -> tuple:
        _, source, destination, port, *_ = event
        return source, destination, (source, destination, port)

    def advance(self, now: int) -> None:
        while self.events and self.events[0][0] < now - self.horizon:
            self._update(self.events.popleft(), -1)

    def _update(self, event: tuple, sign: int) -> None:
        _, source, destination, port, failed, nbytes, packets = event
        for scope, key, peer in zip(
            self.stats, self.keys(event), (destination, source, destination), strict=True
        ):
            if key not in scope:
                scope[key] = [0, Counter(), Counter(), 0, 0.0, 0.0]
            state = scope[key]
            state[0] += sign
            state[1][peer] += sign
            state[2][port] += sign
            if state[1][peer] == 0:
                del state[1][peer]
            if state[2][port] == 0:
                del state[2][port]
            state[3] += sign * failed
            state[4] += sign * nbytes
            state[5] += sign * packets
            if state[0] == 0:
                del scope[key]

    def insert(self, event: tuple) -> None:
        self.events.append(event)
        self._update(event, 1)

    def describe(self, event: tuple) -> list[float]:
        result = []
        for scope, key in zip(self.stats, self.keys(event), strict=True):
            state = scope.get(key)
            if state is None:
                result.extend([0.0] * len(STATISTICS))
            else:
                count, peers, ports, failed, nbytes, packets = state
                result.extend(
                    [
                        np.log1p(count),
                        np.log1p(len(peers)),
                        np.log1p(len(ports)),
                        failed / count,
                        np.log1p(max(0, nbytes)),
                        np.log1p(max(0, packets)),
                    ]
                )
        return result


def describe_events(events: list[tuple], query_rows: np.ndarray) -> np.ndarray:
    """Return query features; input events are observation-only, never labels."""
    output = np.zeros((len(query_rows), len(CONTEXT_FEATURES)), dtype=np.float32)
    positions = {int(row): position for position, row in enumerate(query_rows)}
    order = sorted(range(len(events)), key=lambda index: events[index][0])
    histories = [RollingBehavior(seconds) for seconds in HORIZONS]
    start = 0
    while start < len(order):
        now = events[order[start]][0]
        stop = start + 1
        while stop < len(order) and events[order[stop]][0] == now:
            stop += 1
        for history in histories:
            history.advance(now)
        for row in order[start:stop]:
            if row in positions:
                output[positions[row]] = np.asarray(
                    [value for history in histories for value in history.describe(events[row])]
                )
        for row in order[start:stop]:
            for history in histories:
                history.insert(events[row])
        start = stop
    return output


def extract_day(task: tuple[str, str, str]) -> dict:
    filename, input_path, output_path = task
    with np.load(input_path) as payload:
        dataset = str(payload["dataset"])
    if dataset == "cic":
        # Original unlabelled table also contains rows omitted by labelling.
        frame = pd.read_csv(filename, dtype=str, keep_default_na=False)
        if {"label", "type"}.intersection(frame.columns):
            raise ValueError("CIC context must read the original unlabelled table")
    else:
        observation_fields = (
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
        )
        frame = pd.read_csv(
            filename,
            dtype=str,
            keep_default_na=False,
            usecols=lambda name: XIIOTID_COLUMNS.get(name, name) in observation_fields,
        ).rename(columns=XIIOTID_COLUMNS)
        for name in observation_fields:
            if name not in frame:
                frame[name] = "-"
        # Match preparation's column order, excluding both label columns at IO.
    start_time = pd.to_numeric(frame["ts"], errors="coerce").to_numpy()
    valid_time = np.isfinite(start_time)
    skipped_invalid_time = int((~valid_time).sum())
    frame = frame.loc[valid_time].copy()
    start_time = start_time[valid_time]
    signature_frame = frame.copy()
    for name in ("src_ip_bytes", "dst_ip_bytes", "missed_bytes", "http_uri"):
        if name not in signature_frame:
            signature_frame[name] = "-"
    signature = pd.util.hash_pandas_object(signature_frame, index=False).to_numpy()
    with np.load(input_path) as payload:
        wanted = set(payload["signature"].tolist())
    chosen = np.flatnonzero(np.isin(signature, list(wanted)))
    duration = pd.to_numeric(frame["duration"], errors="coerce").fillna(0).clip(lower=0)
    times = np.ceil(start_time + duration.to_numpy()).astype(np.int64)
    sources = pd.factorize(pd.concat((frame["src_ip"], frame["dst_ip"])), sort=True)[0]
    source, destination = sources[: len(frame)], sources[len(frame) :]
    ports = pd.to_numeric(frame["dst_port"], errors="coerce").fillna(-1).astype(int).to_numpy()
    nbytes = sum(
        pd.to_numeric(frame[name], errors="coerce").fillna(0).clip(lower=0)
        for name in ("src_bytes", "dst_bytes")
    )
    packets = sum(
        pd.to_numeric(frame[name], errors="coerce").fillna(0).clip(lower=0)
        for name in ("src_pkts", "dst_pkts")
    )
    failure = frame["conn_state"].isin(FAILED_STATES).astype(int).to_numpy()
    events = list(zip(times, source, destination, ports, failure, nbytes, packets, strict=True))
    values = describe_events(events, chosen)
    # Duplicate coarse signatures have the same keys and completed second;
    # strict-second history must consequently be identical.
    unique, first, inverse = np.unique(signature[chosen], return_index=True, return_inverse=True)
    if not np.array_equal(values, values[first][inverse]):
        raise ValueError("Aliased connection signature has inconsistent causal context")
    np.savez_compressed(output_path, signature=unique, context=values[first])
    return {
        "day": Path(filename).stem,
        "events": len(events),
        "unique_queries": len(unique),
        "skipped_invalid_time": skipped_invalid_time,
    }


def run(args: argparse.Namespace) -> None:
    source, output, root = (
        Path(args.data).resolve(),
        Path(args.output).resolve(),
        Path(args.root).resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    with np.load(source) as payload:
        data = {key: payload[key] for key in payload.files}
    dataset = str(data["dataset"])
    if dataset == "cic":
        paths = [
            root / f"data/processed/cic_ids2017_conn/{day}.csv"
            for day in ("Monday", "Thursday", "Friday")
        ]
    elif dataset == "xiiotid":
        paths = [root / "data/X-IIoTID/X-IIoTID dataset.csv"]
    else:
        raise ValueError("This preparation supports CIC and X-IIoTID tables")
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "input_sha256": sha256(source),
            "context_sources_sha256": {p.name: sha256(p) for p in paths},
            "status": "anonymous_strictly_prior_connection_context",
            "history": "all unlabelled completed connections; no current-second or future events",
            "identity": "addresses only transient counter keys; never numeric model input",
            "horizons_seconds": HORIZONS,
        },
    )
    tasks = [(str(p), str(source), str(output / f"{p.stem}_context.npz")) for p in paths]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(extract_day, tasks))
    wanted = {int(key): i for i, key in enumerate(data["signature"])}
    values = np.zeros((len(wanted), len(CONTEXT_FEATURES)), dtype=np.float32)
    matched = np.zeros(len(wanted), dtype=bool)
    for path in paths:
        with np.load(output / f"{path.stem}_context.npz") as payload:
            for key, row in zip(payload["signature"], payload["context"], strict=True):
                position = wanted[int(key)]
                values[position], matched[position] = row, True
    if not matched.all():
        raise ValueError(f"Missing context for {int((~matched).sum())} prepared rows")
    data["features"] = np.concatenate((data["features"], values), axis=1)
    data["feature_names"] = np.concatenate((data["feature_names"], np.asarray(CONTEXT_FEATURES)))
    corruption_groups(data["feature_names"])
    np.savez_compressed(output / "records.npz", **data)
    write_json(
        output / "report.json",
        {
            "days": reports,
            "records": len(values),
            "dimensions": data["features"].shape[1],
            "output_sha256": sha256(output / "records.npz"),
        },
    )
    print(reports, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
