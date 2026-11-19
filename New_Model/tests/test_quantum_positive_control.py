import numpy as np

from fedxmag_next.quantum_positive_control import (
    best_of,
    classical_kernels,
    fit_kernel_ridge,
    positive_control,
    predict,
    rbf,
    relation_predictability,
)


def test_kernel_ridge_recovers_smooth_function_and_selects_on_validation():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 2))
    y = np.sin(x[:, :1]) + 0.01 * rng.normal(size=(300, 1))
    model = fit_kernel_ridge(rbf(0.5), x[:200], y[:200], x[200:250], y[200:250])
    mse = float(((predict(model, x[250:]) - y[250:]) ** 2).mean())
    assert mse < 0.01
    best = best_of(classical_kernels(x[:200]), x[:200], y[:200], x[200:250], y[200:250])
    assert best["name"].startswith("rbf")


def test_positive_control_runs_and_reports_every_family():
    out = positive_control("quantum_projected_g1", n_anchor=60, n_train=120, n_val=60,
                           n_cal=60, n_test=120, seed=3)
    assert set(out["families"]) == {"quantum_target", "quantum_all", "product_circuit",
                                    "classical_suite"}
    for row in out["families"].values():
        assert 0.0 <= row["auroc"] <= 1.0 and 0.0 <= row["fpr"] <= 1.0
    assert out["geometric_difference"]["minimum_g"] > 0


def test_relation_predictability_mean_only_reference():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(240, 8))
    y = np.c_[z[:, 0] * z[:, 1], z[:, 2]]
    out = relation_predictability(z[:120], y[:120], z[120:180], y[120:180], z[180:], y[180:])
    assert out["families"]["classical_suite"]["r2_vs_mean"] > 0.3
    assert {"quantum", "product_circuit", "classical_suite"} <= set(out["families"])
