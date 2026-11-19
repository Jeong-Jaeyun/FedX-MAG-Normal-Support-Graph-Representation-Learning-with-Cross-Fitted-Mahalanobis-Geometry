"""Transductive graph baseline: novelty relative to verified normal endpoints/pairs.

This explicitly uses address EQUALITY in a trusted graph reference. It is not an
identity-free neural detector. No address is interpreted as a numeric feature.
Only complete verified-normal windows update the bounded post-query memory.
Frozen reference scores protect calibration, so adaptive alarms are a subset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from .detector import conformal_threshold
from .experiments import sha256, write_json


class TrustedPairs:
    def __init__(self, source_pairs, capacity):
        if capacity <= 0:
            raise ValueError("Memory capacity must be positive")
        self.source_pairs = set(source_pairs)
        self.source_nodes = {node for pair in self.source_pairs for node in pair}
        self.capacity = capacity
        self.recent = deque()
        self.nodes, self.pairs = Counter(), Counter()

    def score(self, pairs):
        source = []
        destination = []
        novel_pairs = []
        for pair in pairs:
            a, b = pair
            source.append(a not in self.source_nodes and self.nodes[a] == 0)
            destination.append(b not in self.source_nodes and self.nodes[b] == 0)
            novel_pairs.append(pair not in self.source_pairs and self.pairs[pair] == 0)
        source, destination = np.asarray(source), np.asarray(destination)
        return {
            "new_source_fraction": float(source.mean()),
            "new_destination_fraction": float(destination.mean()),
            "new_endpoint_fraction": float((source | destination).mean()),
            "new_pair_fraction": float(np.mean(novel_pairs)),
        }

    def update(self, pairs):
        for pair in pairs:
            if len(self.recent) == self.capacity:
                old = self.recent.popleft()
                self.pairs[old] -= 1
                if not self.pairs[old]:
                    del self.pairs[old]
                for node in old:
                    self.nodes[node] -= 1
                    if not self.nodes[node]:
                        del self.nodes[node]
            self.recent.append(pair)
            self.pairs[pair] += 1
            self.nodes.update(pair)


def run(args):
    output = Path(args.output).resolve()
    data_path = Path(args.data).resolve()
    control_path = Path(args.control).resolve()
    model_path = Path(args.detector).resolve()
    with np.load(data_path) as data, np.load(args.keys) as keys:
        if not np.array_equal(data["signature"], keys["signature"]):
            raise ValueError("Grouping keys refer to different observations")
        minutes = data["observed_at"] // 60
        signature = data["signature"]
        labels = data["label"]
        dataset = str(data["dataset"])
        src, dst = keys["source"], keys["destination"]
    registration = json.loads((control_path / "registration.json").read_text())
    if registration["data_sha256"] != sha256(data_path):
        raise ValueError("Whole-window control refers to different observations")
    if registration["detector_sha256"] != sha256(model_path):
        raise ValueError("Source roles differ from the control source model")
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    reference = state["source_role_indices"]["train"].numpy()
    reference = reference[np.argsort(signature[reference])[: args.max_reference]]
    if np.any(labels[reference] != 0):
        raise ValueError("An attack entered the trusted reference")
    stored = torch.load(control_path / "scores.pt", weights_only=True)
    target = stored["minutes"].numpy()
    if minutes[reference].max() >= target.min():
        raise ValueError("Trusted source graph contains future observations")
    source_minutes = stored["methods"]["raw_knn"]["source_calibration_minutes"].numpy()
    order = np.argsort(minutes, kind="stable")
    unique, starts, counts = np.unique(minutes[order], return_index=True, return_counts=True)
    groups = {
        int(t): order[start : start + count]
        for t, start, count in zip(unique, starts, counts, strict=True)
    }
    pairs = [(int(a), int(b)) for a, b in zip(src, dst, strict=True)]
    prior = [pairs[int(i)] for i in reference]
    frozen = TrustedPairs(prior, args.memory_pairs)
    adapted = TrustedPairs(prior, args.memory_pairs)
    names = tuple(frozen.score(prior))
    history = {
        name: deque(
            (
                frozen.score([pairs[int(i)] for i in groups[int(t)]])[name]
                for t in source_minutes[-args.buffer :]
            ),
            maxlen=args.buffer,
        )
        for name in names
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": dataset,
            "data_sha256": sha256(data_path),
            "source_detector_sha256": sha256(model_path),
            "whole_window_scores_sha256": sha256(control_path / "scores.pt"),
            "keys_sha256": sha256(Path(args.keys)),
            "census_sha256": registration["census_sha256"],
            "fit": "same capped source-train normal rows; address equality explicitly used",
            "feedback": "whole-normal verification after query; calibration always frozen scores",
            "limits": "transductive identity reference baseline, not identity-free neural learning",
        },
    )
    values = {
        f"{name}_{suffix}": np.zeros(len(target))
        for name in names
        for suffix in ("frozen", "adaptive")
    }
    flags = {name: np.zeros(len(target), dtype=bool) for name in values}
    valid = np.zeros(len(target), dtype=bool)
    truth = stored["labels"].numpy()
    audit = []
    for i, minute in enumerate(target):
        if int(minute) not in groups:
            continue
        observations = [pairs[int(j)] for j in groups[int(minute)]]
        base, current = frozen.score(observations), adapted.score(observations)
        for name in names:
            if current[name] > base[name] + 1e-12:
                raise AssertionError("Trusted graph extension increased novelty")
            threshold = conformal_threshold(history[name], 0.05)
            for suffix, score in (("frozen", base[name]), ("adaptive", current[name])):
                method = f"{name}_{suffix}"
                values[method][i] = score
                if threshold is not None:
                    flags[method][i] = score > threshold
                    valid[i] = True
        audit.append({"minute": int(minute), "previous_normal_pairs": len(adapted.recent)})
        if truth[i] == 0:
            adapted.update(observations)
            for name in names:
                history[name].append(base[name])
    methods = {}
    for name, decision in flags.items():
        normal, attack = valid & (truth == 0), valid & (truth == 1)
        entry = {
            "tpr": float(decision[attack].mean()),
            "fpr": float(decision[normal].mean()),
            "coverage": float(valid.mean()),
            "normal_windows": int(normal.sum()),
            "attack_windows": int(attack.sum()),
            "end_to_end_attack_recall": float(decision[truth == 1].mean()),
            "per_type_tpr": {},
        }
        for kind in sorted({k for types in stored["types"] for k in types}):
            selected = np.array([kind in types for types in stored["types"]])
            entry["per_type_tpr"][kind] = {
                "available_windows": int(selected.sum()),
                "tpr": float(decision[selected & valid].mean()),
            }
        methods[name] = entry
        print(name, {key: entry[key] for key in ("tpr", "fpr", "coverage")}, flush=True)
    for name in names:
        if np.any(flags[f"{name}_adaptive"] & ~flags[f"{name}_frozen"]):
            raise AssertionError("Trusted extension created an alarm")
    write_json(
        output / "report.json",
        {
            "status": "trusted_adjacency_reference_development",
            "dataset": dataset,
            "methods": methods,
            "source_nodes": len(frozen.source_nodes),
            "source_pairs": len(frozen.source_pairs),
            "memory_audit": audit,
            "new_alarms_from_adaptation": 0,
        },
    )
    torch.save(
        {
            "minutes": stored["minutes"],
            "labels": stored["labels"],
            "types": stored["types"],
            "methods": {
                name: {
                    "energy": torch.from_numpy(values[name]),
                    "flags": torch.from_numpy(decision),
                    "valid": torch.from_numpy(valid),
                }
                for name, decision in flags.items()
            },
            "source_pairs": sorted(frozen.source_pairs),
            "recent_verified_pairs": list(adapted.recent),
        },
        output / "decisions.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "keys", "detector", "control", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--buffer", type=int, default=39)
    parser.add_argument("--memory-pairs", type=int, default=5000)
    parser.add_argument("--max-reference", type=int, default=20000)
    run(parser.parse_args())
