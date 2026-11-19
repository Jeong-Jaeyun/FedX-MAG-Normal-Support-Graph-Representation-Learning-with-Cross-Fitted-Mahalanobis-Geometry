"""Paired test of vetted target-normal support and neural adaptation.

Per target capture, the first 20% of available normal windows extend references,
the next 20% calibrate, and only subsequent windows are evaluated. Captures with
fewer than ten normal windows remain explicitly unsupported. The oracle normal
status is a dataset-supplied research assumption, not autonomous discovery.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .capture_experiment import GROUPS, node_refit
from .contrastive_energy import DependencyEnergy, train
from .experiments import aggregate_rows, fuse, knn_scores, load_roles, metrics, write_json
from .flow_experiment import signed_scale


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.workers)
    root, output, source = (
        Path(args.root).resolve(),
        Path(args.output).resolve(),
        Path(args.source).resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    roles, paths = load_roles(root)
    windows = [window for role in roles for window in roles[role]]
    cached = torch.load(
        root / "output/root_cause_20261026/flow_context_cache.pt", weights_only=True
    )
    flows = [row for role in roles for row in cached["features"][role]]
    source_scores = torch.load(source / "scores.pt", weights_only=True)
    scaler = torch.load(paths["train"].parent / "standardizer.pt", weights_only=True)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "status": "paired_vetted_normal_commissioning_development",
            "target_prefix": (
                "First 20% reference, next 20% calibration; captures with >=10 normals"
            ),
            "evaluation": "Same suffix normal and attack windows for every paired method",
            "normal_status": (
                "oracle labels from experimental dataset; external verification in deployment"
            ),
            "neural_update": (
                "fine tune on source + repeated target-reference normals; source selection only"
            ),
        },
    )
    report = {"status": "paired_vetted_normal_commissioning_development", "folds": {}}
    archive = {}
    started = time.perf_counter()
    for fold, original in source_scores.items():
        indices = original["indices"]
        captures = defaultdict(list)
        for index in indices["normal"]:
            captures[windows[index].metadata["session_key"]].append(index)
        target_reference, target_calibration, target_normal, target_attack = [], [], [], []
        cutoffs, unsupported = {}, []
        for capture, members in captures.items():
            ordered = sorted(members, key=lambda i: windows[i].timestamp())
            if len(ordered) < 10:
                unsupported.append(capture)
                continue
            part = int(0.2 * len(ordered))
            target_reference.extend(ordered[:part])
            target_calibration.extend(ordered[part : 2 * part])
            target_normal.extend(ordered[2 * part :])
            cutoffs[capture] = windows[ordered[2 * part - 1]].timestamp()
        for index in indices["attack"]:
            capture = windows[index].metadata["session_key"]
            if capture in cutoffs and windows[index].timestamp() > cutoffs[capture]:
                target_attack.append(index)
        selected = {
            "selection": indices["selection"],
            "calibration": target_calibration,
            "normal": target_normal,
            "attack": target_attack,
        }
        if len(target_calibration) < 19 or not target_normal or not target_attack:
            report["folds"][fold] = {
                "status": "insufficient_target_support",
                "unsupported": unsupported,
            }
            continue
        node_rows = node_refit(windows, indices["train"], scaler)
        state = torch.load(source / f"fold_{fold}_model.pt", weights_only=True)
        mean, scale = state["mean"], state["scale"]
        flow_rows = [(row.float() - mean) / scale for row in flows]
        all_reference = indices["train"] + target_reference
        channels = []
        for rows in (node_rows, flow_rows):
            reference = torch.cat([rows[index] for index in all_reference])
            channel = {}
            for role, members in selected.items():
                query = [rows[index] for index in members]
                channel[role] = aggregate_rows(knn_scores(reference, torch.cat(query)), query)
            channels.append(channel)
        commissioned_raw = fuse(channels)
        source_model = DependencyEnergy(len(mean))
        source_model.load_state_dict(state["model"])
        source_model.eval()
        matrices = {
            role: torch.cat([flow_rows[index] for index in members])
            for role, members in selected.items()
        }
        with torch.no_grad():
            neural_source = signed_scale(
                {
                    role: aggregate_rows(
                        source_model(values), [flows[index] for index in selected[role]]
                    )
                    for role, values in matrices.items()
                }
            )
        tuned = copy.deepcopy(source_model)
        source_rows = torch.cat([flow_rows[index] for index in indices["train"]])
        target_rows = torch.cat([flow_rows[index] for index in target_reference])
        repeats = max(1, int(np.ceil(len(source_rows) / len(target_rows))))
        training = torch.cat((source_rows, target_rows.repeat(repeats, 1)))
        print(
            json.dumps(
                {
                    "phase": "commissioning_fold",
                    "fold": fold,
                    "target_reference": len(target_reference),
                    "target_calibration": len(target_calibration),
                    "normal": len(target_normal),
                    "attack": len(target_attack),
                }
            ),
            flush=True,
        )
        history = train(
            tuned, training, matrices["selection"], GROUPS, seed=args.seed, epochs=args.epochs
        )
        with torch.no_grad():
            neural_tuned = signed_scale(
                {
                    role: aggregate_rows(tuned(values), [flows[index] for index in selected[role]])
                    for role, values in matrices.items()
                }
            )
        methods = {
            "commissioned_raw": commissioned_raw,
            "commissioned_raw_plus_source_neural": fuse([commissioned_raw, neural_source]),
            "commissioned_raw_plus_tuned_neural": fuse([commissioned_raw, neural_tuned]),
            "target_calibrated_source_neural": neural_source,
            "target_calibrated_tuned_neural": neural_tuned,
        }
        # Matched subset of the original source-only predictions. Its calibration
        # remains the disjoint source-calibration fold, not a target role.
        for name in ("raw_node_flow_knn", "raw_plus_flow_dependency"):
            stored = original["methods"][name]
            normal_positions = {index: position for position, index in enumerate(indices["normal"])}
            attack_positions = {index: position for position, index in enumerate(indices["attack"])}
            methods[f"source_only_{name}"] = {
                "selection": stored["selection"].numpy(),
                "calibration": stored["calibration"].numpy(),
                "normal": stored["normal"][
                    [normal_positions[index] for index in target_normal]
                ].numpy(),
                "attack": stored["attack"][
                    [attack_positions[index] for index in target_attack]
                ].numpy(),
            }
        results = {
            name: metrics(scores["calibration"], scores["normal"], scores["attack"])
            for name, scores in methods.items()
        }
        entry = {
            "counts": {
                "reference": len(target_reference),
                "calibration": len(target_calibration),
                "normal": len(target_normal),
                "attack": len(target_attack),
            },
            "supported_captures": sorted(cutoffs),
            "unsupported_captures": sorted(unsupported),
            "methods": results,
        }
        report["folds"][fold] = entry
        write_json(output / f"fold_{fold}.json", entry)
        print(
            json.dumps(
                {
                    "fold": fold,
                    "metrics": {
                        name: {key: result[key] for key in ("auroc", "tpr", "fpr")}
                        for name, result in results.items()
                    },
                }
            ),
            flush=True,
        )
        torch.save(
            {"model": tuned.state_dict(), "mean": mean, "scale": scale, "history": history},
            output / f"fold_{fold}_model.pt",
        )
        archive[fold] = {
            "target_reference": target_reference,
            "selected": selected,
            "methods": {
                name: {role: torch.from_numpy(value) for role, value in scores.items()}
                for name, scores in methods.items()
            },
        }
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "report.json", report)
    torch.save(archive, output / "scores.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
