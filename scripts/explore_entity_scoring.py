"""EXPLORATORY (post-test): entity-level vs graph-level normal-support scoring.

Uses frozen M0/M1 checkpoints; nothing is retrained. Every detector is fitted
on train-normal entities, thresholds are split-conformal on calibration-normal,
and the consumed test partition is scored. Because the scoring design was
motivated by a diagnosis of this same test partition, every number here is
exploratory and needs confirmation on an untouched source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fedxmag.config import feature_groups_from_config, load_stage1_config
from fedxmag.confirmation import model_builders
from fedxmag.m1.detection import auroc, cluster_bootstrap_detection, detection_summary
from fedxmag.m1.entity import (
    AGGREGATIONS,
    EntityScorer,
    EntityStates,
    StateGraphModel,
    encode_entities,
    raw_entity_features,
)
from fedxmag.m1.training import clean_view
from fedxmag.models.m0 import _encoder_values, _shuffle_endpoints_within_graph
from fedxmag.schema import collate_graphs
from fedxmag.serialization import load_graph_samples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARED = PROJECT_ROOT / "data" / "processed" / "xiiotid_stage1_v3b_development"
M0_DIR = PROJECT_ROOT / "output" / "m0" / "xiiotid_stage1_v3b_confirmation"
M1_DIR = PROJECT_ROOT / "output" / "m1" / "xiiotid_stage1_m1"
TARGETS = (0.01, 0.05, 0.10)


@torch.no_grad()
def control_states(model: Any, kind: str, samples: list, groups: Any) -> EntityStates:
    """Final entity states of the Set-context and Shuffled-graph controls."""

    model.eval()
    nodes, edges = [], []
    for start in range(0, len(samples), 32):
        batch = collate_graphs(samples[start : start + 32], groups)
        masked = clean_view(batch)
        if kind == "shuffled_graph":
            masked = _shuffle_endpoints_within_graph(
                masked, model.shuffle_seed, model.endpoint_shuffle_mode
            )
            batch = masked.batch
        node_state = model.node_input(_encoder_values(masked, "node"))
        edge_state = model.edge_input(_encoder_values(masked, "edge"))
        for layer in model.layers:
            if kind == "shuffled_graph":
                node_state, edge_state = layer(node_state, edge_state, batch.edge_index)
            else:
                node_state, edge_state = layer(node_state, edge_state, batch.node_graph_index,
                                               batch.edge_graph_index, batch.num_graphs)
        for graph in range(batch.num_graphs):
            nodes.append(node_state[batch.node_graph_index == graph].double())
            edges.append(edge_state[batch.edge_graph_index == graph].double())
    return EntityStates(nodes, edges)


def controls(normal: np.ndarray, attack: np.ndarray, ctx: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    weights, values = [], []
    edges = ctx["size_edges"]
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        n = (ctx["normal_nodes"] >= low) & (ctx["normal_nodes"] < high)
        a = (ctx["attack_nodes"] >= low) & (ctx["attack_nodes"] < high)
        if n.sum() >= 5 and a.sum() >= 5:
            weights.append(n.sum() * a.sum())
            values.append(auroc(normal[n], attack[a]))
    out["size_stratified_auroc"] = float(np.average(values, weights=weights))
    weights, values = [], []
    for date in ctx["mixed_dates"]:
        n, a = ctx["normal_dates"] == date, ctx["attack_dates"] == date
        if n.sum() >= 3 and a.sum() >= 3:
            weights.append(n.sum() * a.sum())
            values.append(auroc(normal[n], attack[a]))
    out["within_date_auroc"] = float(np.average(values, weights=weights))
    for label, low, high in (("fraction<0.1", 0.0, 0.1), ("0.1-0.5", 0.1, 0.5),
                             ("fraction>=0.5", 0.5, 1.01)):
        mask = (ctx["attack_fraction"] >= low) & (ctx["attack_fraction"] < high)
        out[f"auroc[{label}]"] = auroc(normal, attack[mask])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "diagnosis")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_stage1_config(PROJECT_ROOT / "configs" / "stage1_m0_v3_development.yaml")
    groups = feature_groups_from_config(config)
    roles = {
        "fit": "train_normal", "calibration": "calibration_normal",
        "normal": "test_normal", "attack": "test_attack",
    }
    samples = {role: load_graph_samples(PREPARED / f"{name}.pt", groups)
               for role, name in roles.items()}
    windows = {(w["session"], w["window_start_ns"]): w
               for w in json.loads((args.output_dir / "test_band_windows.json").read_text())}
    ctx: dict[str, Any] = {
        "normal_nodes": np.asarray([s.node_features.shape[0] for s in samples["normal"]], float),
        "attack_nodes": np.asarray([s.node_features.shape[0] for s in samples["attack"]], float),
        "normal_dates": np.asarray([s.metadata["session_key"] for s in samples["normal"]]),
        "attack_dates": np.asarray([s.metadata["session_key"] for s in samples["attack"]]),
        "attack_fraction": np.asarray([
            windows[(s.metadata["session_key"], s.metadata["window_start_ns"])]["attack_fraction"]
            for s in samples["attack"]
        ]),
    }
    combined = np.concatenate((ctx["normal_nodes"], ctx["attack_nodes"]))
    ctx["size_edges"] = list(np.unique(np.quantile(combined, np.linspace(0, 1, 5)))[:-1]) + [np.inf]
    ctx["mixed_dates"] = sorted(set(ctx["normal_dates"]) & set(ctx["attack_dates"]))

    builders = model_builders(config, groups)
    representations: dict[str, dict[str, Any]] = {"raw_features": {"deterministic": None}}
    for predictor, seeds in (("graph", (17, 29, 41, 53, 67)), ("set_context", (17, 29, 41, 53, 67)),
                             ("shuffled_graph", (17, 29, 41, 53, 67))):
        representations[f"m0_confirmed_{predictor}"] = {
            str(seed): (predictor, M0_DIR / f"m0_{predictor}_seed_{seed}.pt") for seed in seeds
        }
    for arm in ("m0_recon", "m1_lw"):
        representations[arm] = {
            str(seed): ("graph", M1_DIR / f"m1_{arm}_seed_{seed}.pt") for seed in (17, 29, 41)
        }
    entity_dir = args.output_dir / "entity_m1"
    entity_paths = {str(seed): entity_dir / f"entity_m1_lw_seed_{seed}.pt" for seed in (17, 29, 41)}
    if all(path.exists() for path in entity_paths.values()):
        representations["entity_m1_lw"] = {
            seed: ("graph", path) for seed, path in entity_paths.items()
        }

    results: dict[str, Any] = {}
    scores_for_bootstrap: dict[str, dict[str, np.ndarray]] = {}
    members_by_method: dict[str, list[str]] = {}
    for name, members in representations.items():
        for member, spec in members.items():
            if spec is None:
                states = {role: raw_entity_features(values) for role, values in samples.items()}
            else:
                predictor, path = spec
                state_dict = torch.load(path, map_location="cpu", weights_only=True)["model"]
                if predictor == "graph":
                    model = StateGraphModel(**_graph_kwargs(config, groups))
                    model.load_state_dict(state_dict)
                    states = {role: encode_entities(model, values, groups)
                              for role, values in samples.items()}
                else:
                    model = builders[predictor]()
                    model.load_state_dict(state_dict)
                    states = {role: control_states(model, predictor, values, groups)
                              for role, values in samples.items()}
            for kind in ("node", "edge"):
                scorer = EntityScorer.fit(states["fit"], kind)
                entity_scores = {role: scorer.entity_scores(states[role])
                                 for role in ("calibration", "normal", "attack")}
                for how in AGGREGATIONS:
                    from fedxmag.m1.entity import aggregate

                    graph = {role: aggregate(values, how) for role, values in entity_scores.items()}
                    method = f"{name}|{kind}|{how}"
                    summary = detection_summary(graph["calibration"], graph["normal"],
                                                graph["attack"], TARGETS)
                    summary.update(controls(graph["normal"], graph["attack"], ctx))
                    results.setdefault(method, {})[member] = summary
                    key = f"{method}#{member}"
                    scores_for_bootstrap[key] = graph
                    members_by_method.setdefault(method, []).append(key)
        print("done", name, flush=True)

    # Graph-level pooled references from the sealed M1 test (identical thresholds).
    sealed = np.load(M1_DIR / "test_scores.npz")
    test_report = json.loads((M1_DIR / "test_report.json").read_text())
    for method in ("handcrafted_combined_lw", "raw_attribute_classical", "m0_confirmed_graph",
                   "m1_lw", "m0_recon"):
        label = f"POOLED:{method}"
        for member in test_report["methods"][method]["members"]:
            graph = {role: sealed[f"{method}#{member}#{role}"]
                     for role in ("calibration", "normal", "attack")}
            summary = detection_summary(graph["calibration"], graph["normal"], graph["attack"],
                                        TARGETS)
            summary.update(controls(graph["normal"], graph["attack"], ctx))
            results.setdefault(label, {})[member] = summary
            scores_for_bootstrap[f"{label}#{member}"] = graph
            members_by_method.setdefault(label, []).append(f"{label}#{member}")

    def mean_over(method: str, getter) -> float:
        return float(np.mean([getter(item) for item in results[method].values()]))

    def rate(metric: str, target: float):
        return lambda r: r["at_calibrated_fpr"][f"{target:g}"][metric]

    table = {}
    for method in results:
        table[method] = {
            "auroc": mean_over(method, lambda r: r["auroc"]),
            "auroc_seed_std": float(np.std([r["auroc"] for r in results[method].values()])),
            "auprc": mean_over(method, lambda r: r["auprc"]),
            **{f"tpr@{t:g}": mean_over(method, rate("tpr", t)) for t in TARGETS},
            **{f"fpr@{t:g}": mean_over(method, rate("fpr", t)) for t in TARGETS},
            **{key: mean_over(method, lambda r, key=key: r[key])
               for key in ("size_stratified_auroc", "within_date_auroc", "auroc[fraction<0.1]",
                           "auroc[0.1-0.5]", "auroc[fraction>=0.5]")},
            "members": len(results[method]),
        }
    focus = [
        "entity_m1_lw|node|top3_mean", "entity_m1_lw|node|max",
        "m1_lw|node|top3_mean", "m0_confirmed_graph|node|top3_mean", "m0_recon|node|top3_mean",
        "m0_confirmed_shuffled_graph|node|top3_mean", "m0_confirmed_set_context|node|top3_mean",
        "raw_features|node|top3_mean", "m1_lw|edge|top3_mean",
        "POOLED:m1_lw", "POOLED:m0_confirmed_graph", "POOLED:handcrafted_combined_lw",
    ]
    focus = [method for method in focus if method in members_by_method]
    candidate_pairs = [
        ("entity_m1_lw|node|top3_mean", "m0_recon|node|top3_mean"),
        ("entity_m1_lw|node|top3_mean", "m1_lw|node|top3_mean"),
        ("entity_m1_lw|node|top3_mean", "POOLED:handcrafted_combined_lw"),
        ("m1_lw|node|top3_mean", "POOLED:m1_lw"),
        ("m0_confirmed_graph|node|top3_mean", "POOLED:m0_confirmed_graph"),
        ("m1_lw|node|top3_mean", "POOLED:handcrafted_combined_lw"),
        ("m0_confirmed_graph|node|top3_mean", "POOLED:handcrafted_combined_lw"),
        ("m1_lw|node|top3_mean", "m0_recon|node|top3_mean"),
        ("m1_lw|node|top3_mean", "m0_confirmed_graph|node|top3_mean"),
        ("m0_confirmed_graph|node|top3_mean", "m0_confirmed_shuffled_graph|node|top3_mean"),
        ("m0_confirmed_graph|node|top3_mean", "m0_confirmed_set_context|node|top3_mean"),
        ("m0_confirmed_graph|node|top3_mean", "raw_features|node|top3_mean"),
    ]
    pairs = [(a, b) for a, b in candidate_pairs if a in focus and b in focus]
    clusters = {
        "calibration": [s.metadata["session_key"] for s in samples["calibration"]],
        "normal": [s.metadata["session_key"] for s in samples["normal"]],
        "attack": [s.metadata["session_key"] for s in samples["attack"]],
    }
    bootstrap = cluster_bootstrap_detection(
        {k: v for k, v in scores_for_bootstrap.items() if k.split("#")[0] in focus},
        {m: members_by_method[m] for m in focus}, clusters, pairs, TARGETS,
        args.bootstrap, 20261026,
    )
    report = {
        "status": "exploratory_post_test",
        "note": "Scoring design motivated by diagnosis of this same test partition.",
        "table": table,
        "per_member": results,
        "bootstrap": bootstrap,
    }
    (args.output_dir / "entity_scoring_exploration.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.output_dir / "entity_scoring_scores.npz",
                        **{f"{k}#{role}": v[role] for k, v in scores_for_bootstrap.items()
                           if k.split("#")[0] in focus for role in v})
    columns = ("auroc", "auroc_seed_std", "size_stratified_auroc", "within_date_auroc",
               "auroc[fraction<0.1]", "tpr@0.01", "fpr@0.01", "tpr@0.05", "fpr@0.05")
    print(f"{'method':46s} " + " ".join(f"{c[:12]:>12s}" for c in columns))
    for method in sorted(table, key=lambda m: -table[m]["auroc"]):
        print(f"{method:46s} " + " ".join(f"{table[method][c]:12.3f}" for c in columns))
    for key, entry in bootstrap["differences"].items():
        print(key, {m: tuple(round(v[part], 3) for part in ("mean_difference", "lower", "upper"))
                    for m, v in entry.items()})


def _graph_kwargs(config: dict[str, Any], groups: Any) -> dict[str, Any]:
    model = config["model"]
    return {"node_dim": groups.node_dim, "edge_dim": groups.edge_dim,
            "hidden_dim": int(model["hidden_dim"]), "graph_dim": int(model["graph_dim"]),
            "num_layers": int(model["num_layers"]), "dropout": float(model["dropout"])}


if __name__ == "__main__":
    main()
