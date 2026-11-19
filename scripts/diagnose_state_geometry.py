"""EXPLORATORY: geometry of encoder entity states and anomaly localization.

(1) Train-normal spectra of frozen node/edge states, used only to set the
entity-level guardrails from train-normal statistics. (2) On the consumed test,
whether the highest-energy node of an attack window is an attack endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fedxmag.config import feature_groups_from_config, load_stage1_config
from fedxmag.m1.covariance import spectral_diagnostics
from fedxmag.m1.entity import EntityScorer, StateGraphModel, encode_entities
from fedxmag.serialization import load_graph_samples

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "data" / "processed" / "xiiotid_stage1_v3b_development"


def main() -> None:
    torch.set_num_threads(4)
    config = load_stage1_config(ROOT / "configs" / "stage1_m0_v3_development.yaml")
    groups = feature_groups_from_config(config)
    model_config = config["model"]
    kwargs = {"node_dim": groups.node_dim, "edge_dim": groups.edge_dim,
              "hidden_dim": int(model_config["hidden_dim"]),
              "graph_dim": int(model_config["graph_dim"]),
              "num_layers": int(model_config["num_layers"]),
              "dropout": float(model_config["dropout"])}
    train = load_graph_samples(PREPARED / "train_normal.pt", groups)
    attack = load_graph_samples(PREPARED / "test_attack.pt", groups)
    windows = {(w["session"], w["window_start_ns"]): w for w in json.loads(
        (ROOT / "output" / "diagnosis" / "test_band_windows.json").read_text())}
    report = {}
    for seed in (17, 29, 41, 53, 67):
        model = StateGraphModel(**kwargs)
        path = ROOT / f"output/m0/xiiotid_stage1_v3b_confirmation/m0_graph_seed_{seed}.pt"
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
        train_states = encode_entities(model, train, groups)
        entry = {}
        for kind in ("node", "edge"):
            stacked = train_states.stacked(kind)
            covariance = torch.cov(stacked.T)
            entry[f"{kind}_raw_shape"] = spectral_diagnostics(
                covariance / (torch.trace(covariance) / covariance.shape[0])
            )
            entry[f"{kind}_count"] = int(stacked.shape[0])
        scorer = EntityScorer.fit(train_states, "node")
        attack_states = encode_entities(model, attack, groups)
        hits, base, ranks = [], [], []
        for sample, scores in zip(attack, scorer.entity_scores(attack_states), strict=True):
            info = windows[(sample.metadata["session_key"], sample.metadata["window_start_ns"])]
            endpoints = set(info["attack_endpoint_keys"])
            flags = np.asarray([key in endpoints for key in sample.metadata["endpoint_keys"]])
            if not flags.any() or flags.all():
                continue
            order = np.argsort(-scores.numpy())
            hits.append(bool(flags[order[0]]))
            base.append(float(flags.mean()))
            ranks.append(int(np.flatnonzero(flags[order])[0]) + 1)
        entry["localization"] = {
            "windows": len(hits),
            "top1_is_attack_endpoint": float(np.mean(hits)),
            "random_node_is_attack_endpoint": float(np.mean(base)),
            "median_rank_of_first_attack_endpoint": float(np.median(ranks)),
        }
        report[str(seed)] = entry
        print(seed, json.dumps(entry["localization"]), flush=True)
    (ROOT / "output" / "diagnosis" / "state_geometry.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
