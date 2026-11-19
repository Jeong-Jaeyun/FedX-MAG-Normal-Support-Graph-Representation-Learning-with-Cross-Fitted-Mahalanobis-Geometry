"""Simulate what the cross-fitted Mahalanobis energy rewards, independent of data.

For Gaussian embeddings with a controlled spectrum (lambda_i proportional to
exp(-gamma * i), trace fixed), estimate the expected symmetric cross-fitted
energy / d under each covariance estimator at the in-batch reference sizes used
by M1. A loss that is flat in the spectrum carries no first-order signal about
geometry; a loss that falls as the effective rank falls rewards collapse.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from fedxmag.m1.covariance import ESTIMATORS, estimate_covariance, mahalanobis_energy

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def effective_rank(spectrum: torch.Tensor) -> float:
    probabilities = spectrum / spectrum.sum()
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--reference-sizes", type=int, nargs="+", default=[32, 64, 256])
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output" / "m1" / "simulation_ns_incentive.json",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(args.seed)
    dimension = args.dimension
    gammas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0]
    results = []
    for reference in args.reference_sizes:
        for gamma in gammas:
            spectrum = torch.exp(-gamma * torch.arange(dimension, dtype=torch.float64))
            spectrum = spectrum * dimension / spectrum.sum()
            energies = {estimator: [] for estimator in ESTIMATORS}
            energies["empirical_in_sample"] = []
            for _ in range(args.replicates):
                draws = torch.randn(
                    (2 * reference, dimension), generator=generator, dtype=torch.float64
                ) * spectrum.sqrt()
                first, second = draws[:reference], draws[reference:]
                for estimator in ESTIMATORS:
                    values = [
                        mahalanobis_energy(query, estimate_covariance(fit, estimator)).mean()
                        for fit, query in ((first, second), (second, first))
                    ]
                    energies[estimator].append(float(sum(values) / 2) / dimension)
                if reference > dimension:
                    in_sample = estimate_covariance(first, "empirical", spectral_floor=0.0)
                    energies["empirical_in_sample"].append(
                        float(mahalanobis_energy(first, in_sample).mean()) / dimension
                    )
            summary = {}
            for name, values in energies.items():
                if not values:
                    continue
                tensor = torch.tensor(values, dtype=torch.float64)
                summary[name] = {
                    "mean": float(tensor.mean()),
                    "median": float(tensor.median()),
                    "standard_error": float(tensor.std() / math.sqrt(tensor.numel())),
                }
            results.append(
                {
                    "reference_size": reference,
                    "gamma": gamma,
                    "effective_rank": effective_rank(spectrum),
                    "energy_over_d": summary,
                }
            )
            print(json.dumps({"n_R": reference, "gamma": gamma,
                              "r_eff": round(effective_rank(spectrum), 2),
                              **{k: round(v["mean"], 3) for k, v in summary.items()}}),
                  flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "dimension": dimension,
                "replicates": args.replicates,
                "seed": args.seed,
                "spectrum": "lambda_i proportional to exp(-gamma i), trace = d",
                "hotelling_expectation_over_d": {
                    str(reference): (reference + 1) * (reference - 1)
                    / (reference * (reference - dimension - 2))
                    for reference in args.reference_sizes
                    if reference > dimension + 2
                },
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
