import numpy as np
import pytest

from fedxmag_next.quantum_features import ProjectedQuantumMap
from fedxmag_next.quantum_gates import (
    approximate_dimension,
    fidelity_kernel,
    geometric_difference,
    jacobian_readability,
    linear_kernel,
    local_bloch_length,
    magnitude_response,
    memory_capacity,
    minimum_classical_geometric_difference,
    rbf_kernel,
    trace_normalize,
)


def test_magnitude_identity_passes_and_bounded_map_fails():
    identity = magnitude_response(lambda z: z, 4, reference=1024, directions=64)
    assert identity["passes"] and identity["monotone_beyond_typical"]
    assert identity["detection_radius"] <= 3 * identity["typical_radius"]
    bounded = magnitude_response(np.tanh, 4, reference=1024, directions=64)
    assert not bounded["passes"]
    assert bounded["median_energy_over_normal_q95"][-1] < 50


def test_jacobian_of_linear_map_matches_singular_values():
    a = np.array([[2.0, 0.0], [0.0, 0.5], [0.0, 0.0]])
    out = jacobian_readability(lambda z: z @ a.T, np.zeros((3, 2)))
    assert out["sigma_min_mean"] == pytest.approx(0.5, rel=1e-6)
    assert out["sigma_max_mean"] == pytest.approx(2.0, rel=1e-6)


def test_product_circuit_has_unit_bloch_length():
    z = np.random.default_rng(1).normal(size=(32, 4))
    features = ProjectedQuantumMap(qubits=4, entangled=False).transform(z).numpy()
    assert local_bloch_length(features, 4)["mean"] == pytest.approx(1.0, abs=1e-10)


def test_memory_capacity_shift_register_and_null_map():
    rng = np.random.default_rng(2)
    h_tr, h_te = rng.normal(size=(600, 4, 3)), rng.normal(size=(300, 4, 3))
    full = memory_capacity(h_tr.reshape(600, -1), h_te.reshape(300, -1), h_tr, h_te)
    assert full["capacity"] == pytest.approx(4.0, abs=1e-2)
    null = memory_capacity(rng.normal(size=(600, 5)), rng.normal(size=(300, 5)), h_tr, h_te)
    assert null["capacity"] < 0.1


def test_geometric_difference_identity_and_separation():
    x = np.random.default_rng(3).normal(size=(60, 3))
    k = rbf_kernel(x, 0.5)
    same = geometric_difference(k, k)
    assert same["g"] is not None and 0.9 < same["g"] <= 1.0 + 1e-9
    # Orthogonal "quantum" states versus a nearly rank-one classical kernel.
    far = geometric_difference(np.eye(60), np.ones((60, 60)) + 1e-3 * np.eye(60))
    assert far["grid"][0]["g_gen"] > 5


def test_dimension_bounds_and_trace_normalisation():
    assert approximate_dimension(np.eye(20)) == pytest.approx(20.0)
    assert approximate_dimension(np.ones((20, 20))) == pytest.approx(1.0)
    assert np.trace(trace_normalize(3 * np.eye(5))) == pytest.approx(5.0)


def test_classical_suite_contains_linear_and_gaussian_kernels():
    x = np.random.default_rng(4).normal(size=(40, 2))
    out = minimum_classical_geometric_difference(linear_kernel(x) + 1e-6 * np.eye(40), x)
    assert out["closest_classical_kernel"] is not None
    assert out["minimum_g"] <= out["by_kernel"]["rbf_1"] + 1e-12


def test_fidelity_kernel_is_unit_diagonal_for_pure_states():
    q = ProjectedQuantumMap(qubits=3)
    s = q.statevectors(np.random.default_rng(5).normal(size=(10, 3)))
    assert np.allclose(np.diag(fidelity_kernel(s)), 1.0)
