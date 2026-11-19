"""Describe fixed hard votes on already verified heterogeneous predictions."""

import json
from pathlib import Path

import numpy as np
import torch

from fedxmag_next.complementary_views import metrics
from fedxmag_next.experiments import sha256, write_json

ROOT = Path("New_Model/runs")
OUTPUT = ROOT / "round232_heterogeneous_hard_votes"
RUNS = {
    "cic": "round227_cic_heterogeneous_consensus",
    "ton": "round230_ton_heterogeneous_consensus",
    "xiiotid": "round231_xiiotid_heterogeneous_consensus",
}


def run():
    if OUTPUT.exists():
        raise FileExistsError("Never overwrite a completed heterogeneous vote audit")
    # Require every planned completed report before creating any audit output.
    for folder in RUNS.values():
        for suffix in (Path(), Path("capacity50")):
            report = json.loads((ROOT / folder / suffix / "report.json").read_text())
            if not report["all_single_member_flags_and_coverage_exact"]:
                raise ValueError("Hard vote audit requires verified component controls")
    OUTPUT.mkdir()
    provenance, all_reports, all_predictions = {}, {}, {}
    for dataset, folder in RUNS.items():
        all_reports[dataset], all_predictions[dataset] = {}, {}
        for capacity in (39, 50):
            source = ROOT / folder
            if capacity == 50:
                source = source / "capacity50"
            path = source / "decisions.pt"
            stored = torch.load(path, weights_only=True)
            provenance[f"{dataset}_{capacity}"] = sha256(path)
            labels = stored["labels"].numpy()
            members = ("neural", "forest", "gaussian")
            flags = np.column_stack(
                [stored["methods"][f"control_{name}"]["flags"].numpy() for name in members]
            )
            valid = np.logical_and.reduce(
                [stored["methods"][f"control_{name}"]["valid"].numpy() for name in members]
            )
            original = json.loads((source / "report.json").read_text())
            for index, name in enumerate(members):
                repeated = metrics(flags[:, index], valid, labels, stored["types"])
                if repeated != original["controls"][name]:
                    raise ValueError("Hard vote audit changed an original component metric")
            designs = {
                "neural_forest_both": flags[:, 0] & flags[:, 1],
                "neural_forest_either": flags[:, 0] | flags[:, 1],
                "neural_gaussian_both": flags[:, 0] & flags[:, 2],
                "neural_gaussian_either": flags[:, 0] | flags[:, 2],
                "three_majority": flags.sum(axis=1) >= 2,
                "three_all": flags.all(axis=1),
                "three_any": flags.any(axis=1),
            }
            reports, predictions = {}, {}
            for name, values in designs.items():
                reports[name] = metrics(values, valid, labels, stored["types"])
                predictions[name] = torch.from_numpy(values)
                print(dataset, capacity, name, {k: reports[name][k] for k in ("tpr", "fpr")})
            all_reports[dataset][str(capacity)] = {
                "controls": original["controls"],
                "component_error_overlap": original["component_error_overlap"],
                "methods": reports,
            }
            all_predictions[dataset][str(capacity)] = {
                "minutes": stored["minutes"],
                "labels": stored["labels"],
                "types": stored["types"],
                "valid": torch.from_numpy(valid),
                "methods": predictions,
            }
    write_json(
        OUTPUT / "report.json",
        {
            "status": "fixed_hard_vote_development_audit",
            "inputs_sha256": provenance,
            "query_before_feedback_inherited_from_verified_components": True,
            "weights_or_thresholds_fitted_to_target": False,
            "warning": "low FPR alone does not meet the95%/5% success criterion",
            "datasets": all_reports,
        },
    )
    torch.save(all_predictions, OUTPUT / "decisions.pt")


if __name__ == "__main__":
    run()
