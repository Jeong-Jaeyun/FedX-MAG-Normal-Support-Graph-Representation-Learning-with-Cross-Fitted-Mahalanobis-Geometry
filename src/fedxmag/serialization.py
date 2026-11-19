"""Versioned, schema-checked persistence for preprocessed graph samples."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from fedxmag.schema import DEFAULT_FEATURE_GROUPS, FeatureGroups, GraphSample

ARTIFACT_VERSION = 4
SUPPORTED_ARTIFACT_VERSIONS = frozenset({3, ARTIFACT_VERSION})


def schema_fingerprint(groups: FeatureGroups = DEFAULT_FEATURE_GROUPS) -> str:
    schema = {
        "node": groups.node_names,
        "edge": groups.edge_names,
    }
    encoded = json.dumps(schema, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_graph_samples(
    path: str | Path,
    samples: list[GraphSample],
    *,
    overwrite: bool = False,
    groups: FeatureGroups = DEFAULT_FEATURE_GROUPS,
) -> None:
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite graph artifact: {destination}")
    records = []
    for sample in samples:
        sample.validate(groups)
        records.append(
            {
            "node_features": sample.node_features.cpu(),
            "edge_features": sample.edge_features.cpu(),
            "edge_index": sample.edge_index.cpu(),
            "node_available": sample.node_available.cpu(),
            "edge_available": sample.edge_available.cpu(),
            "label": sample.label,
                "metadata_json": json.dumps(sample.metadata, sort_keys=True),
            }
        )
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "schema_fingerprint": schema_fingerprint(groups),
        "node_features": list(groups.node_names),
        "edge_features": list(groups.edge_names),
        "records": records,
    }
    torch.save(payload, destination)


def load_graph_samples(
    path: str | Path, groups: FeatureGroups = DEFAULT_FEATURE_GROUPS
) -> list[GraphSample]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("artifact_version") not in SUPPORTED_ARTIFACT_VERSIONS
    ):
        raise ValueError(
            f"Graph artifact must use one of versions {sorted(SUPPORTED_ARTIFACT_VERSIONS)}"
        )
    if payload.get("schema_fingerprint") != schema_fingerprint(groups):
        raise ValueError("Graph artifact feature schema does not match this code version")
    if tuple(payload.get("node_features", ())) != groups.node_names:
        raise ValueError("Graph artifact node feature order is incompatible")
    if tuple(payload.get("edge_features", ())) != groups.edge_names:
        raise ValueError("Graph artifact edge feature order is incompatible")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Graph artifact records must be a list")
    samples: list[GraphSample] = []
    required = {
        "node_features",
        "edge_features",
        "edge_index",
        "node_available",
        "edge_available",
        "label",
        "metadata_json",
    }
    for index, item in enumerate(records):
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError(f"Invalid graph record at index {index}")
        record = dict(item)
        metadata_json = record.pop("metadata_json")
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid metadata at graph record {index}") from error
        if not isinstance(metadata, dict):
            raise ValueError(f"Metadata at graph record {index} must be an object")
        sample = GraphSample(**record, metadata=metadata)
        sample.validate(groups)
        samples.append(sample)
    return samples
