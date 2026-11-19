import io

import numpy as np
import pytest
import torch

from fedxmag_next.h1_classical_features import ClassicalFeatureMap, SourcePCA
from fedxmag_next.h1_experiment import replay
from fedxmag_next.quantum_features import ProjectedQuantumMap


def kron_at(gate, j, q):
    full = np.array([[1]], dtype=complex)
    for k in range(q):
        full = np.kron(full, gate if k == j else np.eye(2))
    return full


def dense_oracle(z, model):
    q = len(z)
    state = np.eye(2**q, dtype=complex)[:, 0]
    a = 2 * np.arctan(z)
    for layer in range(model.layers):
        for j in range(q):
            for angle, axis in (
                (a[j], "y"),
                (a[(j + 1) % q], "z"),
                (model.angles[layer, j, 0], "y"),
                (model.angles[layer, j, 1], "z"),
            ):
                c, s = np.cos(angle / 2), np.sin(angle / 2)
                g = (
                    np.array([[c, -s], [s, c]])
                    if axis == "y"
                    else np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)])
                )
                state = kron_at(g, j, q) @ state
        if model.entangled:
            for j, k in {tuple(sorted((j, (j + 1) % q))) for j in range(q)}:
                gate = np.eye(2**q)
                for index in range(2**q):
                    bits = format(index, f"0{q}b")
                    if bits[j] == bits[k] == "1":
                        gate[index, index] = -1
                state = gate @ state
    paulis = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    obs = [kron_at(p, j, q) for p in paulis for j in range(q)]
    obs += [kron_at(paulis[2], j, q) @ kron_at(paulis[2], (j + 1) % q, q) for j in range(q)]
    return state, np.array([np.vdot(state, o @ state).real for o in obs])


@pytest.mark.parametrize("q", [2, 3, 4])
@pytest.mark.parametrize("entangled", [False, True])
def test_dense_oracle(q, entangled):
    model = ProjectedQuantumMap(qubits=q, entangled=entangled, chunk_size=2)
    z = np.random.default_rng(18).normal(size=(5, q))
    expected = [dense_oracle(row, model) for row in z]
    np.testing.assert_allclose(
        model.statevectors(z), np.stack([e[0] for e in expected]), atol=1e-13
    )
    np.testing.assert_allclose(model.transform(z), np.stack([e[1] for e in expected]), atol=1e-13)
    np.testing.assert_allclose((np.abs(model.statevectors(z)) ** 2).sum(1), 1, atol=1e-13)


def test_quantum_roundtrip_chunks_shots():
    model = ProjectedQuantumMap(qubits=3, chunk_size=1)
    z = torch.randn(4, 3, generator=torch.Generator().manual_seed(1))
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    restored = ProjectedQuantumMap.from_state_dict(torch.load(buf, weights_only=True))
    torch.testing.assert_close(model.transform(z), restored.transform(z), atol=1e-13, rtol=1e-13)
    measured = model.transform_shots(z, shots=100000, seed=3)
    torch.testing.assert_close(measured, model.transform(z), atol=0.012, rtol=0)
    assert torch.equal(measured, model.transform_shots(z, shots=100000, seed=3))
    with pytest.raises(ValueError):
        model.transform(torch.zeros(3, 4))


@pytest.mark.parametrize(
    "kind", ["linear", "trigonometric", "rff", "random_mlp", "rbf_nystrom", "polynomial_nystrom"]
)
def test_classical_normal_fit_reload(kind):
    x = torch.randn(100, 12, generator=torch.Generator().manual_seed(13))
    before = x.clone()
    adapter = SourcePCA.fit(x)
    z = adapter.transform(x)
    torch.testing.assert_close(z.mean(0), torch.zeros(8, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(
        z.std(0, unbiased=False), torch.ones(8, dtype=torch.float64), atol=1e-12, rtol=0
    )
    mapping = ClassicalFeatureMap.fit(kind, z, landmarks=32)
    buf = io.BytesIO()
    torch.save({"adapter": adapter.state_dict(), "map": mapping.state_dict()}, buf)
    buf.seek(0)
    saved = torch.load(buf, weights_only=True)
    repeated = ClassicalFeatureMap.from_state_dict(saved["map"]).transform(
        SourcePCA.from_state_dict(saved["adapter"]).transform(x)
    )
    torch.testing.assert_close(mapping.transform(z), repeated)
    assert torch.equal(before, x)
    assert torch.isfinite(repeated).all()


def test_feedback_only_affects_later_predictions():
    minutes = np.arange(100, 130)
    energy = np.linspace(0, 30, 30)
    source = np.arange(50, dtype=float) / 5
    labels = np.ones(30, dtype=int)
    altered = labels.copy()
    altered[10] = 0
    a, ta = replay(energy, source, minutes, labels, "immediate")
    b, tb = replay(energy, source, minutes, altered, "immediate")
    np.testing.assert_array_equal(a[:11], b[:11])
    np.testing.assert_array_equal(ta[:11], tb[:11])
    _, tf = replay(energy, source, minutes, altered, "fixed")
    assert np.all(tf == tf[0])


def test_delayed_feedback_respects_arrival():
    minutes = np.arange(100, 300)
    energy = np.full(200, 100.0)
    labels = np.ones(200, dtype=int)
    chosen = np.flatnonzero(np.random.default_rng(731).random(200) < 0.1)
    labels[chosen[0]] = 0
    _, th = replay(energy, np.arange(50, dtype=float), minutes, labels, "delayed_10pct")
    assert np.all(th[: chosen[0] + 5] == th[0])
    assert th[chosen[0] + 5] > th[0]
