"""EXPLORATORY DIAGNOSIS (after the sealed M1 test): where is the attack signal?

Uses the consumed test partition only (plus train-normal as the normal
reference). Answers three questions:

1. Do attack-involved edges and nodes differ from normal entities in the v3b
   behavioural features at all?
2. Does graph-level mean pooling hide an entity-level signal? Compares
   graph-level Mahalanobis on pooled embeddings with entity-level Mahalanobis
   (raw features, M0/M1 encoder states) aggregated by max and top-k.
3. How much of any result survives controls for graph size, capture date, and
   attack-record fraction?

Every detector is fitted on train-normal only. Nothing here is a confirmatory
result.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fedxmag.config import feature_groups_from_config, load_stage1_config
from fedxmag.confirmation import model_builders
from fedxmag.m1.detection import MahalanobisDetector, auroc
from fedxmag.m1.training import clean_view
from fedxmag.models import M0GraphModel
from fedxmag.models.m0 import _encoder_values
from fedxmag.schema import GraphSample, collate_graphs
from fedxmag.serialization import load_graph_samples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARED = PROJECT_ROOT / "data" / "processed" / "xiiotid_stage1_v3b_development"


def load(name: str, groups: Any) -> list[GraphSample]:
    return load_graph_samples(PREPARED / f"{name}.pt", groups)


@torch.no_grad()
def encoder_states(model: M0GraphModel, samples: list[GraphSample], groups: Any,
                   batch_size: int = 32) -> tuple[list[np.ndarray], list[np.ndarray],
                                                  np.ndarray]:
    """Per-graph clean-view node states, edge states, and pooled graph embeddings."""

    model.eval()
    nodes, edges, pooled = [], [], []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        batch = collate_graphs(chunk, groups)
        masked = clean_view(batch)
        node_state = model.node_input(_encoder_values(masked, "node"))
        edge_state = model.edge_input(_encoder_values(masked, "edge"))
        for layer in model.layers:
            node_state, edge_state = layer(node_state, edge_state, batch.edge_index)
        pooled.append(model(masked).graph_embedding.double().numpy())
        for graph in range(batch.num_graphs):
            nodes.append(node_state[batch.node_graph_index == graph].double().numpy())
            edges.append(edge_state[batch.edge_graph_index == graph].double().numpy())
    return nodes, edges, np.concatenate(pooled)


def raw_entities(samples: list[GraphSample], kind: str) -> list[np.ndarray]:
    result = []
    for sample in samples:
        values = getattr(sample, f"{kind}_features").double()
        available = getattr(sample, f"{kind}_available").double()
        result.append(torch.cat((values * available, available), dim=1).numpy())
    return result


def aggregate(scores: list[np.ndarray], how: str) -> np.ndarray:
    output = []
    for values in scores:
        if values.size == 0:
            output.append(0.0)
        elif how == "max":
            output.append(float(values.max()))
        elif how == "top3_mean":
            output.append(float(np.sort(values)[-3:].mean()))
        else:
            output.append(float(values.mean()))
    return np.asarray(output)


def controlled_aurocs(normal: np.ndarray, attack: np.ndarray, context: dict[str, Any]) -> dict:
    """Overall AUROC plus size-stratified, mixed-date, and attack-fraction views."""

    result = {"overall": auroc(normal, attack)}
    edges = context["size_edges"]
    weights, values = [], []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        n_mask = (context["normal_nodes"] >= low) & (context["normal_nodes"] < high)
        a_mask = (context["attack_nodes"] >= low) & (context["attack_nodes"] < high)
        if n_mask.sum() >= 5 and a_mask.sum() >= 5:
            weights.append(n_mask.sum() * a_mask.sum())
            values.append(auroc(normal[n_mask], attack[a_mask]))
    result["size_stratified"] = float(np.average(values, weights=weights)) if values else None
    pairs, aucs = [], []
    for date in context["mixed_dates"]:
        n_mask = context["normal_dates"] == date
        a_mask = context["attack_dates"] == date
        if n_mask.sum() >= 3 and a_mask.sum() >= 3:
            pairs.append(n_mask.sum() * a_mask.sum())
            aucs.append(auroc(normal[n_mask], attack[a_mask]))
    result["within_date"] = float(np.average(aucs, weights=pairs)) if aucs else None
    for label, low, high in (("fraction<0.1", 0.0, 0.1), ("0.1<=fraction<0.5", 0.1, 0.5),
                             ("fraction>=0.5", 0.5, 1.01)):
        mask = (context["attack_fraction"] >= low) & (context["attack_fraction"] < high)
        result[label] = auroc(normal, attack[mask]) if mask.sum() >= 5 else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "diagnosis")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_stage1_config(PROJECT_ROOT / "configs" / "stage1_m0_v3_development.yaml")
    groups = feature_groups_from_config(config)
    train = load("train_normal", groups)
    test_normal = load("test_normal", groups)
    test_attack = load("test_attack", groups)
    windows = {
        (item["session"], item["window_start_ns"]): item
        for item in json.loads((args.output_dir / "test_band_windows.json").read_text())
    }

    # Attack-involvement flags from the raw records of each test-attack window.
    edge_flags, node_flags, fractions, attack_dates = [], [], [], []
    for sample in test_attack:
        info = windows[(sample.metadata["session_key"], sample.metadata["window_start_ns"])]
        pairs = {tuple(pair) for pair in info["attack_pair_keys"]}
        pairs |= {(b, a) for a, b in pairs}
        endpoints = set(info["attack_endpoint_keys"])
        edge_flags.append(np.asarray([tuple(e) in pairs for e in sample.metadata["edge_keys"]]))
        node_flags.append(np.asarray([n in endpoints for n in sample.metadata["endpoint_keys"]]))
        fractions.append(info["attack_fraction"])
        attack_dates.append(sample.metadata["session_key"])
    normal_dates = np.asarray([s.metadata["session_key"] for s in test_normal])
    attack_dates_array = np.asarray(attack_dates)
    normal_nodes = np.asarray([s.node_features.shape[0] for s in test_normal], float)
    attack_nodes = np.asarray([s.node_features.shape[0] for s in test_attack], float)
    context = {
        "size_edges": np.unique(np.quantile(np.concatenate((normal_nodes, attack_nodes)),
                                            np.linspace(0, 1, 5))).tolist()[:-1] + [np.inf],
        "normal_nodes": normal_nodes,
        "attack_nodes": attack_nodes,
        "normal_dates": normal_dates,
        "attack_dates": attack_dates_array,
        "mixed_dates": sorted(set(normal_dates) & set(attack_dates_array)),
        "attack_fraction": np.asarray(fractions),
    }
    report: dict[str, Any] = {
        "status": "exploratory_post_test_diagnosis",
        "attack_involved_edge_share": float(np.mean(np.concatenate(edge_flags))),
        "attack_involved_node_share": float(np.mean(np.concatenate(node_flags))),
        "mixed_dates": len(context["mixed_dates"]),
    }

    # 1. Feature-level entity separability (standardized v3b features).
    feature_level: dict[str, Any] = {}
    for kind, flags, names in (("edge", edge_flags, groups.edge_names),
                               ("node", node_flags, groups.node_names)):
        for column, name in enumerate(names):
            def collect(samples, masks=None, keep=True, kind=kind, column=column):
                values = []
                for index, sample in enumerate(samples):
                    available = getattr(sample, f"{kind}_available")[:, column].numpy()
                    feature = getattr(sample, f"{kind}_features")[:, column].numpy()
                    rows = available.copy()
                    if masks is not None:
                        rows &= masks[index] if keep else ~masks[index]
                    values.append(feature[rows])
                return np.concatenate(values) if values else np.empty(0)

            normal = collect(test_normal)
            involved = collect(test_attack, flags, True)
            background = collect(test_attack, flags, False)
            if min(normal.size, involved.size, background.size) < 20:
                continue
            feature_level[f"{kind}:{name}"] = {
                "involved_vs_normal_auroc": auroc(normal, involved),
                "background_vs_normal_auroc": auroc(normal, background),
                "involved_vs_background_auroc": auroc(background, involved),
                "involved_available": int(involved.size),
            }
    report["feature_level"] = feature_level

    # 2. Scorers: graph-level pooled energy vs entity-level energy with aggregation.
    scorers: dict[str, Any] = {}

    def entity_scorer(name: str, train_entities, normal_entities, attack_entities, flags):
        reference = torch.from_numpy(np.concatenate(train_entities))
        detector = MahalanobisDetector.fit(reference, "ledoit_wolf")
        score = lambda items: [  # noqa: E731
            detector.score(torch.from_numpy(item)).numpy() if len(item) else np.empty(0)
            for item in items
        ]
        normal_scores, attack_scores = score(normal_entities), score(attack_entities)
        entity = auroc(np.concatenate(normal_scores),
                       np.concatenate([s[f] for s, f in zip(attack_scores, flags, strict=True)]))
        entry = {"entity_level_involved_vs_normal_auroc": entity}
        for how in ("mean", "max", "top3_mean"):
            entry[f"graph_{how}"] = controlled_aurocs(
                aggregate(normal_scores, how), aggregate(attack_scores, how), context
            )
        scorers[name] = entry

    entity_scorer("raw_edge_features", raw_entities(train, "edge"),
                  raw_entities(test_normal, "edge"), raw_entities(test_attack, "edge"), edge_flags)
    entity_scorer("raw_node_features", raw_entities(train, "node"),
                  raw_entities(test_normal, "node"), raw_entities(test_attack, "node"), node_flags)

    model_config = config["model"]
    common = {"node_dim": groups.node_dim, "edge_dim": groups.edge_dim,
              "hidden_dim": int(model_config["hidden_dim"]),
              "graph_dim": int(model_config["graph_dim"]),
              "num_layers": int(model_config["num_layers"]),
              "dropout": float(model_config["dropout"])}
    checkpoints = [
        ("m0_confirmed_graph", seed,
         PROJECT_ROOT / f"output/m0/xiiotid_stage1_v3b_confirmation/m0_graph_seed_{seed}.pt")
        for seed in (17, 29, 41)
    ] + [
        ("m1_lw", seed, PROJECT_ROOT / f"output/m1/xiiotid_stage1_m1/m1_m1_lw_seed_{seed}.pt")
        for seed in (17, 29, 41)
    ]
    for name, seed, path in checkpoints:
        model = model_builders(config, groups)["graph"]() if name.startswith("m0") else (
            M0GraphModel(**common)
        )
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
        train_nodes, train_edges, train_pooled = encoder_states(model, train, groups)
        normal_nodes_state, normal_edges_state, normal_pooled = encoder_states(
            model, test_normal, groups
        )
        attack_nodes_state, attack_edges_state, attack_pooled = encoder_states(
            model, test_attack, groups
        )
        pooled_detector = MahalanobisDetector.fit(torch.from_numpy(train_pooled), "ledoit_wolf")
        scorers[f"{name}#{seed}#pooled_graph_embedding"] = {
            "graph_pooled": controlled_aurocs(
                pooled_detector.score(torch.from_numpy(normal_pooled)).numpy(),
                pooled_detector.score(torch.from_numpy(attack_pooled)).numpy(),
                context,
            )
        }
        entity_scorer(f"{name}#{seed}#edge_states", train_edges, normal_edges_state,
                      attack_edges_state, edge_flags)
        entity_scorer(f"{name}#{seed}#node_states", train_nodes, normal_nodes_state,
                      attack_nodes_state, node_flags)
        print("done", name, seed, flush=True)
    report["scorers"] = scorers
    (args.output_dir / "entity_diagnosis.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({k: v for k, v in report.items() if k not in ("feature_level", "scorers")}))
    print("feature-level (|AUROC-0.5| sorted):")
    for key, entry in sorted(feature_level.items(),
                             key=lambda item: -abs(item[1]["involved_vs_normal_auroc"] - 0.5)):
        print(f"  {key:48s} inv-vs-norm {entry['involved_vs_normal_auroc']:.3f}  "
              f"bg-vs-norm {entry['background_vs_normal_auroc']:.3f}  "
              f"inv-vs-bg {entry['involved_vs_background_auroc']:.3f}  "
              f"n={entry['involved_available']}")
    print("scorers:")
    summary = defaultdict(list)
    for key, entry in scorers.items():
        base = key.split("#")[0] + "#" + key.split("#")[-1] if "#" in key else key
        summary[base].append(entry)
    for base, entries in summary.items():
        line = [f"  {base:40s}"]
        if "entity_level_involved_vs_normal_auroc" in entries[0]:
            entity = np.mean([e["entity_level_involved_vs_normal_auroc"] for e in entries])
            line.append(f"entity {entity:.3f}")
        for agg in ("graph_pooled", "graph_mean", "graph_max", "graph_top3_mean"):
            if agg in entries[0]:
                parts = []
                for view in ("overall", "size_stratified", "within_date", "fraction<0.1",
                             "fraction>=0.5"):
                    values = [e[agg][view] for e in entries if e[agg][view] is not None]
                    parts.append(f"{view}={np.mean(values):.3f}" if values else f"{view}=na")
                line.append(f"{agg}[{' '.join(parts)}]")
        print(" ".join(line))


if __name__ == "__main__":
    main()
