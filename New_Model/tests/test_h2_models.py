"""Independent forward and derivative checks for the H2 representation family."""

import numpy as np
import pytest
import torch

from fedxmag_next.h2_models import ClassicalMLPEnergy, ReuploadingQuantumEnergy
from fedxmag_next.quantum_features import ProjectedQuantumMap


@pytest.mark.parametrize("qubits", [2, 3, 8])
@pytest.mark.parametrize("entangled", [False, True])
def test_h1_exact_forward_and_batch_equivalence(qubits, entangled):
    z = torch.randn(5, qubits, generator=torch.Generator().manual_seed(91), dtype=torch.float64)
    h1 = ProjectedQuantumMap(qubits=qubits, layers=2, seed=29, entangled=entangled)
    h2 = ReuploadingQuantumEnergy(qubits=qubits, layers=2, seed=29, entangled=entangled)
    actual = h2.embed(z)
    torch.testing.assert_close(actual, h1.transform(z), atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(actual, torch.cat([h2.embed(row[None]) for row in z]),
                               atol=2e-14, rtol=2e-14)
    state = h2.statevectors(z)
    np.testing.assert_allclose(state.detach().numpy(), h1.statevectors(z), atol=2e-14)
    torch.testing.assert_close(state.abs().square().sum(1), torch.ones(5, dtype=torch.float64))
    assert actual.shape == (5, 4 * qubits)
    assert actual.abs().max() <= 1 + 1e-14


def _dense_oracle(z, angles, entangled):
    """Separate dense Kronecker construction, with four unfused local gates."""
    qubits = len(z)
    state = np.zeros(2**qubits, dtype=np.complex128)
    state[0] = 1
    a = 2 * np.arctan(z)

    def gate(j, angle, kind):
        nonlocal state
        if kind == "y":
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            local = np.array([[c, -s], [s, c]])
        else:
            local = np.diag(np.exp(np.array([-0.5j, 0.5j]) * angle))
        full = np.array([[1]])
        for k in range(qubits):
            full = np.kron(full, local if k == j else np.eye(2))
        state = full @ state

    for layer in angles:
        for j in range(qubits):
            for theta, kind in ((a[j], "y"), (a[(j + 1) % qubits], "z"),
                                (layer[j, 0], "y"), (layer[j, 1], "z")):
                gate(j, theta, kind)
        if entangled:
            for j, k in sorted({tuple(sorted((i, (i + 1) % qubits)))
                                for i in range(qubits)}):
                diagonal = [(-1 if (n >> (qubits-j-1)) & 1
                             and (n >> (qubits-k-1)) & 1 else 1)
                            for n in range(2**qubits)]
                state = np.diag(diagonal) @ state
    return state


@pytest.mark.parametrize("entangled", [False, True])
def test_dense_oracle_with_changed_trainable_angles(entangled):
    model = ReuploadingQuantumEnergy(qubits=3, layers=3, seed=41, entangled=entangled)
    with torch.no_grad():
        model.angles.add_(torch.linspace(-0.2, 0.3, model.angles.numel()).reshape_as(model.angles))
    z = torch.tensor([[0.3, -0.2, 1.2]], dtype=torch.float64)
    expected = _dense_oracle(z[0].numpy(), model.angles.detach().numpy(), entangled)
    np.testing.assert_allclose(model.statevectors(z)[0].detach().numpy(), expected, atol=2e-14)


@pytest.mark.parametrize("entangled", [False, True])
def test_parameter_and_input_gradients_finite_difference(entangled):
    model = ReuploadingQuantumEnergy(qubits=3, layers=2, entangled=entangled)
    z = torch.tensor([[0.13, -0.7, 1.1], [-0.4, 0.21, 0.8]], dtype=torch.float64,
                     requires_grad=True)
    loss = model(z).square().sum()
    loss.backward()
    for tensor, index in ((model.angles, (0, 0, 0)), (model.angles, (1, 2, 1)),
                          (model.readout.weight, (0, 4)), (model.readout.bias, (0,)),
                          (z, (0, 1))):
        analytical = tensor.grad[index].item()
        original = tensor[index].item()
        epsilon = 1e-6
        with torch.no_grad():
            tensor[index] = original + epsilon
            plus = model(z).square().sum().item()
            tensor[index] = original - epsilon
            minus = model(z).square().sum().item()
            tensor[index] = original
        numerical = (plus - minus) / (2 * epsilon)
        assert analytical == pytest.approx(numerical, abs=2e-9, rel=2e-5)
    assert torch.isfinite(model.angles.grad).all()
    assert model.angles.grad.abs().sum() > 0


def test_serialization_empty_float32_and_parameter_budgets(tmp_path):
    model = ReuploadingQuantumEnergy()
    assert sum(p.numel() for p in model.parameters()) == 65
    assert sum(p.numel() for p in ClassicalMLPEnergy(hidden=6, depth=1).parameters()) == 61
    z = torch.ones(4, 8, dtype=torch.float64)
    path = tmp_path / "h2.pt"
    torch.save(model.state_dict(), path)
    restored = ReuploadingQuantumEnergy(seed=999)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(model(z), restored(z), atol=0, rtol=0)
    assert model.embed(z[:0]).shape == (0, 32)
    assert model(z[:0]).shape == (0,)
    single = ReuploadingQuantumEnergy(dtype=torch.float32)
    torch.testing.assert_close(single.embed(z).double(), model.embed(z), atol=2e-6, rtol=2e-6)
    single(z).sum().backward()
    assert single.angles.grad.dtype == torch.float32


def test_determinism_classical_api_and_input_validation():
    before = torch.random.get_rng_state().clone()
    a, b = ClassicalMLPEnergy(seed=17), ClassicalMLPEnergy(seed=17)
    quantum = ReuploadingQuantumEnergy(seed=17)
    assert torch.equal(before, torch.random.get_rng_state())
    z = torch.ones(3, 8, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(a(z), b(z), atol=0, rtol=0)
    assert a.embed(z).shape == (3, 32)
    a(z).sum().backward()
    assert torch.isfinite(z.grad).all()
    for model in (a, quantum):
        with pytest.raises(ValueError):
            model(torch.ones(3, 7))
        with pytest.raises(ValueError):
            model(torch.full((3, 8), torch.nan))
    with pytest.raises(ValueError):
        ReuploadingQuantumEnergy(qubits=1)
