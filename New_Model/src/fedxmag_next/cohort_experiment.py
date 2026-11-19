"""Paired service-conditioned normal support and two-sided typicality study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cohort_energy import CohortEnergyBank, CohortEnergyDetector
from .context_association import cohort_ids
from .daily_detector import DailyDeepEnergyDetector
from .experiments import sha256, write_json
from .window_replay import run as evaluate_windows


@torch.no_grad()
def run(args):
    torch.set_num_threads(args.workers)
    output, path, artifact, control = map(
        lambda p: Path(p).resolve(), (args.output, args.data, args.detector, args.control)
    )
    detector = DailyDeepEnergyDetector.load(artifact)
    registration = json.loads((control / "registration.json").read_text())
    if registration["detector_sha256"] != sha256(artifact):
        raise ValueError("Control belongs to another frozen encoder")
    if registration["aggregation"] != "top3":
        raise ValueError("Paired cohort experiment requires top3 control")
    data_hash = sha256(path)
    if (
        data_hash != detector.state["source_data_sha256"]
        or data_hash != registration["data_sha256"]
    ):
        raise ValueError("Inputs differ from the frozen model/control")
    with np.load(path) as payload:
        data = {key: payload[key] for key in payload.files}
    if tuple(data["feature_names"]) != detector.feature_names:
        raise ValueError("Cohort experiment feature schema differs")
    stored = torch.load(control / "scores.pt", weights_only=True)
    roles = {key: ids.numpy() for key, ids in detector.state["source_role_indices"].items()}
    if any(np.any(data["label"][ids] != 0) for ids in roles.values()):
        raise ValueError("Source geometry received attack rows")
    if any(
        data["observed_at"][ids].max() >= int(stored["minutes"].min()) * 60
        for ids in roles.values()
    ):
        raise ValueError("A source role entered the target time interval")
    if len(np.unique(np.concatenate(list(roles.values())))) != sum(map(len, roles.values())):
        raise ValueError("Source roles overlap")
    reference = roles["train"]
    reference = reference[np.argsort(data["signature"][reference])[: args.max_reference]]
    selection = roles["selection"]
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "registration.json",
        {
            "args": vars(args),
            "dataset": str(data["dataset"]),
            "data_sha256": data_hash,
            "detector_sha256": sha256(artifact),
            "control_scores_sha256": sha256(control / "scores.pt"),
            "census_sha256": registration["census_sha256"],
            "query_labels": "unchanged whole-table census from identical paired control",
            "fit": "frozen encoder; source-train geometry; normal selection quantile units",
            "routing": "observed protocol/service; >=32 reference AND selection rows else global",
            "factorial": "global/cohort geometry x upper/two-sided energy; raw/deep1/8",
            "calibration": "unchanged source minutes and prior verified whole-normal feedback39",
            "quantiles": "q05/q50/q95 score units; not probability or exchangeability guarantees",
            "score_style": args.score_style,
            "empirical_rank_contract": (
                "inclusive selection tails with1e-6relative/1e-9absolute numerical ties; "
                "clipped2min tail for two-sided score; "
                "separate whole-window calibration; no time-exchangeability guarantee"
            ),
        },
    )
    features = torch.from_numpy(data["features"])
    r, s = (
        detector.normalize_features(features[reference]),
        detector.normalize_features(features[selection]),
    )
    ri, si = cohort_ids(features[reference]), cohort_ids(features[selection])
    encodings = {"raw": (r, s)}
    for model, item in zip(detector.models, detector.state["seed_models"], strict=True):
        er, es = model.layers[:-1](r), model.layers[:-1](s)
        encodings[f"seed_{item['seed']}"] = tuple(
            (values - item["embedding_mean"]) / item["embedding_scale"] for values in (er, es)
        )
    banks = {}
    for name, (ref, selected) in encodings.items():
        for modes in (1, 8):
            bank = CohortEnergyBank.fit(ref, selected, ri, si, modes)
            banks[f"{name}_m{modes}"] = bank.entries
            print("fitted", name, modes, "cohorts", list(bank.entries), flush=True)
    state = {
        "format_version": 1,
        "base_encoder": detector.state,
        "banks": banks,
        "score_style": args.score_style,
        "source_reference_indices": torch.from_numpy(reference.copy()),
    }
    torch.save(state, output / "cohort_detector.pt")
    fitted = CohortEnergyDetector(state)
    row_scores = fitted.score(features)
    restored = CohortEnergyDetector.load(output / "cohort_detector.pt")
    audit_rows = np.linspace(0, len(features) - 1, 128, dtype=np.int64)
    reproduced = restored.score(features[audit_rows])
    audit = {}
    for name, expected in row_scores.items():
        expected = expected[audit_rows]
        error = float(((reproduced[name] - expected).abs() / (1 + expected.abs())).max())
        if error > 1e-5:
            raise ValueError(f"Restored cohort inference differs: {name}, {error}")
        audit[name] = error
    minutes = data["observed_at"] // 60
    order = np.argsort(minutes, kind="stable")
    unique, starts, counts = np.unique(minutes[order], return_index=True, return_counts=True)
    groups = {
        int(t): order[start : start + count]
        for t, start, count in zip(unique, starts, counts, strict=True)
    }
    source_minutes = stored["methods"]["raw_knn"]["source_calibration_minutes"].numpy()
    target_minutes = stored["minutes"].numpy()
    membership = np.zeros(len(minutes), dtype=bool)
    membership[roles["calibration"]] = True
    for minute in source_minutes:
        if not membership[groups[int(minute)]].all():
            raise ValueError("A source calibration minute mixes roles")
    archive = dict(stored["methods"])
    for name, energy in row_scores.items():
        energy = energy.numpy()

        def aggregate(minute, values=energy):
            return float(np.sort(values[groups[int(minute)]])[-3:].mean())

        archive[name] = {
            "energy": torch.tensor(
                [aggregate(t) if int(t) in groups else 0.0 for t in target_minutes],
                dtype=torch.float64,
            ),
            "observed": torch.tensor([int(t) in groups for t in target_minutes]),
            "source_calibration": torch.tensor(
                [aggregate(t) for t in source_minutes], dtype=torch.float64
            ),
            "source_calibration_minutes": torch.from_numpy(source_minutes.copy()),
        }
    torch.save({**stored, "methods": archive}, output / "scores.pt")
    evaluate_windows(
        argparse.Namespace(
            run=str(output),
            output=str(output / "evaluation"),
            buffer=args.buffer,
            fpr=0.05,
            history_policy="continuous",
        )
    )
    decisions = torch.load(output / "evaluation/decisions.pt", weights_only=True)
    for name, values in stored["methods"].items():
        for key in ("flags", "valid"):
            if not torch.equal(decisions[name][key], values[key]):
                raise ValueError(f"Paired control replay differs: {name}/{key}")
    report = json.loads((output / "evaluation/report.json").read_text())
    write_json(
        output / "report.json",
        {**report, "all_frozen_control_decisions_exact": True, "restore_scaled_errors": audit},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--buffer", type=int, default=39)
    parser.add_argument("--max-reference", type=int, default=20000)
    parser.add_argument("--score-style", choices=("quantile", "rank"), default="quantile")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
