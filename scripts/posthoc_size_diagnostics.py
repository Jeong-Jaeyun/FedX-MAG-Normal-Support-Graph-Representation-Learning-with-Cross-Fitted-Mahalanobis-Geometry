"""POST HOC, EXPLORATORY: graph-size confounding in the sealed M1 test.

Written after the sealed test was opened, to explain why every learned
representation ranked attack windows below test-normal windows. It reuses the
frozen test scores, refits nothing, and cannot change any registered result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from fedxmag.config import feature_groups_from_config, load_stage1_config
from fedxmag.m1.detection import auroc, spearman
from fedxmag.serialization import load_graph_samples

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "handcrafted_combined_lw",
    "handcrafted_structural_lw",
    "raw_attribute_classical",
    "m0_confirmed_graph",
    "m0_recon",
    "m1_lw",
    "m1_oas",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=PROJECT_ROOT / "configs" / "m1_protocol.yaml"
    )
    parser.add_argument(
        "--m1-dir", type=Path, default=PROJECT_ROOT / "output" / "m1" / "xiiotid_stage1_m1"
    )
    parser.add_argument("--bins", type=int, default=4)
    args = parser.parse_args()
    protocol = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    groups = feature_groups_from_config(load_stage1_config(PROJECT_ROOT / protocol["base_config"]))
    prepared = PROJECT_ROOT / protocol["data"]["prepared_dir"]
    roles = protocol["data"]["roles"]
    samples = {
        role: load_graph_samples(prepared / roles[role], groups)
        for role in ("fit", "calibration", "test_normal", "test_attack")
    }

    def sizes(role: str) -> dict[str, np.ndarray]:
        return {
            "nodes": np.asarray([s.node_features.shape[0] for s in samples[role]], float),
            "edges": np.asarray([s.edge_features.shape[0] for s in samples[role]], float),
            "records": np.asarray([s.metadata["event_count"] for s in samples[role]], float),
        }

    size = {role: sizes(role) for role in samples}
    report: dict[str, object] = {
        "status": "post_hoc_exploratory",
        "note": (
            "Computed after the sealed test; descriptive only. Registered decisions "
            "and primary comparisons are unchanged."
        ),
        "size_quantiles": {
            role: {
                key: {q: float(np.quantile(values, float(q))) for q in ("0.25", "0.5", "0.75")}
                for key, values in role_sizes.items()
            }
            for role, role_sizes in size.items()
        },
        "size_only_auroc": {
            key: auroc(size["test_normal"][key], size["test_attack"][key])
            for key in ("nodes", "edges", "records")
        },
    }
    scores = np.load(args.m1_dir / "test_scores.npz")
    test = json.loads((args.m1_dir / "test_report.json").read_text(encoding="utf-8"))
    combined_nodes = np.concatenate((size["test_normal"]["nodes"], size["test_attack"]["nodes"]))
    edges = np.unique(np.quantile(combined_nodes, np.linspace(0, 1, args.bins + 1)))
    bin_index = {
        role: np.clip(np.searchsorted(edges, size[role]["nodes"], side="right") - 1, 0,
                      len(edges) - 2)
        for role in ("test_normal", "test_attack")
    }
    methods: dict[str, object] = {}
    for method in METHODS:
        members = list(test["methods"][method]["members"])
        per_member = []
        for member in members:
            normal = scores[f"{method}#{member}#normal"]
            attack = scores[f"{method}#{member}#attack"]
            strata = []
            for index in range(len(edges) - 1):
                in_normal = normal[bin_index["test_normal"] == index]
                in_attack = attack[bin_index["test_attack"] == index]
                if in_normal.size >= 5 and in_attack.size >= 5:
                    strata.append(
                        {
                            "bin": index,
                            "nodes_range": [float(edges[index]), float(edges[index + 1])],
                            "normal": int(in_normal.size),
                            "attack": int(in_attack.size),
                            "auroc": auroc(in_normal, in_attack),
                        }
                    )
            weights = np.asarray([item["normal"] * item["attack"] for item in strata], float)
            values = np.asarray([item["auroc"] for item in strata], float)
            per_member.append(
                {
                    "auroc": auroc(normal, attack),
                    "spearman_energy_vs_log_nodes_normal": spearman(
                        normal, np.log(size["test_normal"]["nodes"])
                    ),
                    "spearman_energy_vs_log_nodes_attack": spearman(
                        attack, np.log(size["test_attack"]["nodes"])
                    ),
                    "size_stratified_auroc": float(np.sum(weights * values) / np.sum(weights))
                    if weights.size
                    else math.nan,
                    "strata": strata,
                }
            )
        methods[method] = {
            "auroc_mean": float(np.mean([item["auroc"] for item in per_member])),
            "size_stratified_auroc_mean": float(
                np.mean([item["size_stratified_auroc"] for item in per_member])
            ),
            "spearman_energy_vs_log_nodes_normal_mean": float(
                np.mean([item["spearman_energy_vs_log_nodes_normal"] for item in per_member])
            ),
            "spearman_energy_vs_log_nodes_attack_mean": float(
                np.mean([item["spearman_energy_vs_log_nodes_attack"] for item in per_member])
            ),
            "members": dict(zip(members, per_member, strict=True)),
        }
    report["node_bins"] = edges.tolist()
    report["methods"] = methods
    output = args.m1_dir / "posthoc_size_diagnostics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.m1_dir / "posthoc_sizes.npz",
        **{f"{role}#{key}": values for role, role_sizes in size.items()
           for key, values in role_sizes.items()},
    )
    print(json.dumps({key: report[key] for key in ("size_quantiles", "size_only_auroc")}, indent=1))
    for method, entry in methods.items():
        print(method, {k: round(v, 3) for k, v in entry.items() if k != "members"})


if __name__ == "__main__":
    main()
