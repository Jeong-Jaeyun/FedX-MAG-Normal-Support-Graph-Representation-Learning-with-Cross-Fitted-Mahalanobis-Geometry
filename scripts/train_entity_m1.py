"""EXPLORATORY: train the entity-level M1 variant (normal data only).

Same architecture, masking, optimizer, schedule, and checkpoint rule as the
registered M1 arms; only the normal-support term moves from the pooled graph
embedding to the node and edge states, with a raw-shape rank guardrail.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from fedxmag.config import feature_groups_from_config, load_stage1_config
from fedxmag.confirmation import masker_from_config
from fedxmag.losses import FAMILY_KEYS
from fedxmag.m1.entity import EntitySupportSettings, StateGraphModel, train_entity_m1
from fedxmag.serialization import load_graph_samples

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--name", default="entity_m1_lw")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "output" / "diagnosis" / "entity_m1")
    args = parser.parse_args()
    torch.set_num_threads(1)
    protocol = yaml.safe_load((ROOT / "configs" / "m1_protocol.yaml").read_text())
    config = load_stage1_config(ROOT / protocol["base_config"])
    torch.use_deterministic_algorithms(True)
    groups = feature_groups_from_config(config)
    prepared = ROOT / protocol["data"]["prepared_dir"]
    train = load_graph_samples(prepared / "train_normal.pt", groups)
    select = load_graph_samples(prepared / "select_normal.pt", groups)
    model_config = config["model"]
    torch.manual_seed(args.seed)
    model = StateGraphModel(
        groups.node_dim, groups.edge_dim, hidden_dim=int(model_config["hidden_dim"]),
        graph_dim=int(model_config["graph_dim"]), num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
    )
    settings = EntitySupportSettings()
    training = protocol["training"]
    started = time.perf_counter()
    history = train_entity_m1(
        model, train, select, masker_from_config(config, groups), settings,
        epochs=int(training["epochs"]), batch_size=int(training["batch_size"]),
        validation_batch_size=int(training["validation_batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        validation_mask_seeds=[int(item) for item in training["validation_mask_seeds"]],
        loss_weights={key: float(config["loss"][key]) for key in FAMILY_KEYS},
        seed=args.seed, groups=groups,
    )
    values = [item["selection_metric_value"] for item in history]
    best_epoch = values.index(min(values)) + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / f"{args.name}_seed_{args.seed}"
    torch.save({"model": model.state_dict(), "settings": asdict(settings), "seed": args.seed,
                "best_epoch": best_epoch, "status": "exploratory"}, stem.with_suffix(".pt"))
    stem.with_suffix(".json").write_text(json.dumps(
        {"settings": asdict(settings), "best_epoch": best_epoch, "epochs_run": len(history),
         "wall_seconds": time.perf_counter() - started, "history": history}, indent=1))
    print(json.dumps({"seed": args.seed, "best_epoch": best_epoch, "epochs_run": len(history),
                      "best_worst_family": min(values)}))


if __name__ == "__main__":
    main()
