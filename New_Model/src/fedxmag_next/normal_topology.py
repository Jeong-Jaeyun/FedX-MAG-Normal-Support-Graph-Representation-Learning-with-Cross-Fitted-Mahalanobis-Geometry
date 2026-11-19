"""Bounded categorical normal-contact frequencies; equality metadata only."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

VIEWS = ("destination", "service", "peer_service")


def events(row):
    return (row[1], tuple(row[2:]), tuple(row[1:]))


class NormalTopology:
    def __init__(self, rows, last_time, maximum=5000, strength=20.0):
        if maximum <= 0 or strength <= 0 or not len(rows):
            raise ValueError("Normal topology requires a positive source-normal seed and budgets")
        self.maximum, self.strength = maximum, strength
        self.rows = deque()
        self.context = Counter()
        self.global_counts = [Counter() for _ in VIEWS]
        self.local_counts = [Counter() for _ in VIEWS]
        self.last_time, self.pending = int(last_time), None
        self.admit(rows)

    def change(self, row, increment):
        source = row[0]
        self.context[source] += increment
        if self.context[source] == 0:
            del self.context[source]
        for global_count, local_count, event in zip(
            self.global_counts, self.local_counts, events(row), strict=True
        ):
            for counter, key in ((global_count, event), (local_count, (source, event))):
                counter[key] += increment
                if counter[key] == 0:
                    del counter[key]

    def admit(self, rows):
        for row in rows:
            row = tuple(int(value) for value in row)
            if len(self.rows) == self.maximum:
                self.change(self.rows.popleft(), -1)
            self.rows.append(row)
            self.change(row, 1)

    def score(self, rows):
        output = np.empty((len(rows), len(VIEWS)), dtype=np.float64)
        for i, row in enumerate(rows):
            source = row[0]
            denominator = self.context[source] + self.strength
            for j, event in enumerate(events(row)):
                global_count, local_count = self.global_counts[j], self.local_counts[j]
                backoff = (global_count[event] + 1) / (len(self.rows) + len(global_count) + 1)
                probability = (local_count[(source, event)] + self.strength * backoff) / denominator
                output[i, j] = -np.log(probability)
        return output

    def query(self, now, rows, times):
        times = np.asarray(times, dtype=np.int64)
        if self.pending is not None or now <= self.last_time:
            raise ValueError("Finish topology feedback before a strictly later query")
        if times.shape != (len(rows),) or np.any(times < self.last_time) or np.any(times >= now):
            raise ValueError("Topology observations must belong to the current completed interval")
        result = self.score(rows)
        order = np.argsort(times, kind="stable")
        self.pending = (int(now), [rows[i] for i in order])
        return result

    def feedback(self, *, verified_normal):
        if self.pending is None or not isinstance(verified_normal, bool):
            raise ValueError("External normal verification must follow a topology query")
        now, rows = self.pending
        if verified_normal:
            self.admit(rows)
        self.last_time, self.pending = now, None

    def state_dict(self):
        if self.pending is not None:
            raise ValueError("Checkpoint topology after feedback is committed")
        return {
            "format_version": 1,
            "rows": list(self.rows),
            "last_time": self.last_time,
            "maximum": self.maximum,
            "strength": self.strength,
        }

    @classmethod
    def from_state_dict(cls, state):
        if state.get("format_version") != 1:
            raise ValueError("Unsupported normal topology checkpoint")
        return cls(state["rows"], state["last_time"], state["maximum"], state["strength"])
