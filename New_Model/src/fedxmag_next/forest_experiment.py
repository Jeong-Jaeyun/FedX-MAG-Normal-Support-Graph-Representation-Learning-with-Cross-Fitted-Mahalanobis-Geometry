"""Fixed normal-only IsolationForest controls around preserved causal prototypes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from sklearn.ensemble import IsolationForest

from .daily_detector import DailyDeepEnergyDetector
from .experiments import sha256, write_json
from .hierarchical_experiment import minute_groups
from .prototype_guard import SCOPES
from .prototype_temporal import frozen_thresholds, replay_path
from .window_bridge import aggregate_connections
from .window_replay import run as evaluate_windows


def run(args):
    torch.set_num_threads(args.workers)
    prototype, output = Path(args.prototype), Path(args.output)
    registration = json.loads((prototype / "registration.json").read_text())
    risk_root = Path(registration["args"]["run"])
    risk_registration = json.loads((risk_root / "registration.json").read_text())
    data_path, key_path = (Path(risk_registration["args"][k]) for k in ("data", "keys"))
    if (
        sha256(data_path) != registration["data_sha256"]
        or sha256(key_path) != registration["keys_sha256"]
    ):
        raise ValueError("Forest normal-support input observations or keys changed")
    cache = torch.load(risk_root / "row_scores.pt", weights_only=True)
    distances = torch.load(prototype / "prototype_distances.pt", weights_only=True)
    if sha256(risk_root / "row_scores.pt") != registration["row_cache_sha256"] or not torch.equal(
        cache["combined_indices"], distances["combined_indices"]
    ):
        raise ValueError("Forest normal support does not share the exact causal row census")
    initial = torch.load(prototype / "prototype_risk_detector.pt", weights_only=True)
    base = initial["base_encoder"]
    combined = cache["combined_indices"].numpy()
    with np.load(data_path) as payload:
        features, signature = payload["features"], payload["signature"]
        times = payload["observed_at"][combined]
        labels_rows = payload["label"]
        if tuple(payload["feature_names"]) != tuple(base["feature_names"]):
            raise ValueError("Forest normal support feature order changed")
    with np.load(key_path) as payload:
        if not np.array_equal(signature, payload["signature"]):
            raise ValueError("Forest normal support endpoint signatures changed")
        keys = {k: payload[k][combined] for k in ("source", "destination")}
    roles = {k: v.numpy() for k, v in base["source_role_indices"].items()}
    if any(np.any(labels_rows[ids] != 0) for ids in roles.values()):
        raise ValueError("Forest fit/selection/calibration roles include attack rows")
    train = roles["train"][np.argsort(signature[roles["train"]])[:20000]]
    selected = roles["selection"][np.argsort(signature[roles["selection"]])[:20000]]
    if not np.array_equal(train[:5000], initial["source_reference_indices"].numpy()):
        raise ValueError("Forest source reference differs from the preserved normal seed")
    normalizer = DailyDeepEnergyDetector(base)
    training = normalizer.normalize_features(torch.from_numpy(features[train])).numpy()
    selection = normalizer.normalize_features(torch.from_numpy(features[selected])).numpy()
    queries = normalizer.normalize_features(torch.from_numpy(features[combined])).numpy()
    del features, normalizer
    groups = minute_groups(times // 60)
    old = torch.load(risk_root / "scores.pt", weights_only=True)
    template = old["methods"]["raw_mahalanobis_m1"]
    minutes, labels = old["minutes"].numpy(), old["labels"].numpy()
    cal_minutes = template["source_calibration_minutes"].numpy()
    output.mkdir(parents=True, exist_ok=False)
    common = {
        "args": vars(args),
        "dataset": registration["dataset"],
        "data_sha256": registration["data_sha256"],
        "keys_sha256": registration["keys_sha256"],
        "source_detector_sha256": registration["detector_sha256"],
        "prototype_model_sha256": sha256(prototype / "prototype_risk_detector.pt"),
        "sklearn_version": sklearn.__version__,
        "fit": "exact capped source-normal train rows only",
        "source_train_count": len(train),
        "selection_count": len(selected),
        "score_direction": "negative score_samples; no predict/contamination threshold used",
        "aggregation": registration["aggregation"],
        "max_samples": [256, 2048],
    }
    write_json(output / "registration.json", common)
    forest_scores, forest_reference, frozen_models = {}, {}, {}
    for count in (256, 2048):
        name = f"raw_isolation_forest_samples{count}"
        print("forest fit", registration["dataset"], name, len(train), flush=True)
        forest = IsolationForest(
            n_estimators=256,
            max_samples=count,
            max_features=1.0,
            bootstrap=False,
            contamination="auto",
            random_state=17,
            n_jobs=args.workers,
        )
        forest.fit(training)
        path = output / f"{name}.joblib"
        joblib.dump(forest, path)
        row_reference = -forest.score_samples(selection)
        print("forest score", registration["dataset"], name, len(queries), flush=True)
        values = -forest.score_samples(queries)
        ids = np.linspace(0, len(queries) - 1, 128, dtype=np.int64)
        restored = joblib.load(path)
        if not np.array_equal(-restored.score_samples(queries[ids]), values[ids]):
            raise ValueError("Forest saved model changed actual held-out query scores")
        forest_scores[name], forest_reference[name] = values, row_reference
        sorted_reference = np.sort(row_reference)

        def score_window(minute, values=values, sorted_reference=sorted_reference):
            positions = groups.get(int(minute), np.empty(0, dtype=np.int64))
            return (
                aggregate_connections(
                    values[positions],
                    sorted_reference,
                    registration["aggregation"],
                    reference_sorted=True,
                )
                if len(positions)
                else 0.0
            )

        frozen_models[name] = {
            "energy": torch.tensor([score_window(minute) for minute in minutes]),
            "observed": template["observed"].clone(),
            "source_calibration": torch.tensor([score_window(minute) for minute in cal_minutes]),
            "source_calibration_minutes": template["source_calibration_minutes"].clone(),
        }
    torch.save(
        {
            "combined_indices": cache["combined_indices"],
            "row_scores": {k: torch.from_numpy(v) for k, v in forest_scores.items()},
            "row_reference": {k: torch.from_numpy(v) for k, v in forest_reference.items()},
            "source_train_indices": torch.from_numpy(train),
            "source_selection_indices": torch.from_numpy(selected),
        },
        output / "forest_row_scores.pt",
    )
    for capacity in (39, 50):
        source = risk_root if capacity == 39 else risk_root / "capacity50"
        destination = output if capacity == 39 else output / "capacity50"
        if capacity == 50:
            destination.mkdir()
            write_json(destination / "registration.json", {**common, "capacity": capacity})
        checkpoints = destination / "checkpoints"
        checkpoints.mkdir()
        parent = torch.load(source / "scores.pt", weights_only=True)
        archive = {
            name: parent["methods"][name]
            for name in (
                "raw_knn",
                "raw_mahalanobis_m1",
                "deep_mahalanobis_m1_ensemble",
                "deep_mahalanobis_m8_ensemble",
            )
        }
        archive.update(frozen_models)
        diagnostics = {}
        for name, item in frozen_models.items():
            thresholds = frozen_thresholds(
                item["energy"].numpy(),
                item["observed"].numpy(),
                labels,
                item["source_calibration"].tolist(),
                capacity,
            )
            for scope in SCOPES:
                ratios = distances["ratios"][scope]["raw"].numpy()
                for path in ("prototype", "prototype_risk_fixed"):
                    head = f"{name}_continuous_{path}_{scope}"
                    energy, diagnostics[head] = replay_path(
                        forest_scores[name],
                        forest_reference[name],
                        ratios,
                        minutes,
                        groups,
                        keys,
                        times,
                        registration["aggregation"],
                        thresholds,
                        path,
                        scope,
                        checkpoints / f"{head}.pt",
                        continuous_filter=True,
                    )
                    for feedback in ("frozen", "candidate"):
                        archive[f"{head}_{feedback}"] = {
                            **item,
                            "energy": torch.from_numpy(energy.copy()),
                            "feedback_energy": item["energy"].clone()
                            if feedback == "frozen"
                            else torch.from_numpy(energy.copy()),
                        }
        torch.save({**old, "methods": archive}, destination / "scores.pt")
        evaluate_windows(
            argparse.Namespace(
                run=str(destination),
                output=str(destination / "evaluation"),
                buffer=capacity,
                fpr=0.05,
                history_policy="continuous",
            )
        )
        actual = torch.load(destination / "evaluation/decisions.pt", weights_only=True)
        expected = torch.load(source / "evaluation/decisions.pt", weights_only=True)
        for name in (
            "raw_knn",
            "raw_mahalanobis_m1",
            "deep_mahalanobis_m1_ensemble",
            "deep_mahalanobis_m8_ensemble",
        ):
            if any(
                not torch.equal(value, actual[name][key]) for key, value in expected[name].items()
            ):
                raise ValueError("Forest model comparison changed an unchanged parent decision")
        report = json.loads((destination / "evaluation/report.json").read_text())
        write_json(
            destination / "report.json",
            {
                **report,
                "all_parent_decisions_exact": True,
                "forest_actual_save_reload_query_scores_exact": True,
                "path_diagnostics": diagnostics,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    run(parser.parse_args())
