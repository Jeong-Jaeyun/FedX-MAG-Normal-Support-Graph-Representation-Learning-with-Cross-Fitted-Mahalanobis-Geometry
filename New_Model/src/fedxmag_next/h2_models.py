"""Differentiable H2 representations, simulated exactly with PyTorch.

The circuit matches :mod:`quantum_features`: data RY, shifted-data RZ,
trainable RY/RZ, then a unique-edge CZ ring in each layer. Qubit zero is
the most-significant statevector bit. Features are all X, all Y, all Z,
then adjacent ZZ expectations. This classical simulation is not a speedup.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn


def _validate_input(values: Tensor, dimension: int, parameter: Tensor) -> Tensor:
    values = torch.as_tensor(values, dtype=parameter.dtype, device=parameter.device)
    if values.ndim != 2 or values.shape[1] != dimension:
        raise ValueError(f"Expected finite [rows, {dimension}] inputs")
    if not torch.isfinite(values).all():
        raise ValueError("Inputs must be finite")
    return values


def _linear(input_dim: int, output_dim: int, generator: torch.Generator, dtype) -> nn.Linear:
    # nn.Linear's initialization uses the global RNG; isolate and restore that RNG.
    with torch.random.fork_rng(devices=[]):
        layer = nn.Linear(input_dim, output_dim, dtype=dtype)
    bound = 1 / math.sqrt(input_dim)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        layer.bias.uniform_(-bound, bound, generator=generator)
    return layer


class ReuploadingQuantumEnergy(nn.Module):
    """Trainable local circuit and scalar linear readout (65 parameters at 8/2).

    ``dtype`` must be float32 or float64. Inputs follow the module's dtype/device;
    input gradients are preserved. Use normal ``state_dict``/``load_state_dict``
    with the same constructor configuration to restore the full model.
    """

    def __init__(self, qubits=8, layers=2, seed=17, entangled=True, dtype=torch.float64):
        super().__init__()
        if not isinstance(qubits, int) or not 2 <= qubits <= 12:
            raise ValueError("qubits must be an integer between 2 and 12")
        if not isinstance(layers, int) or layers < 1:
            raise ValueError("layers must be a positive integer")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use float32 or float64 circuit parameters")
        self.qubits, self.layers, self.seed = qubits, layers, seed
        self.entangled = bool(entangled)
        self.output_dim = 4 * qubits
        angles = np.random.default_rng(seed).uniform(-np.pi, np.pi, (layers, qubits, 2))
        self.angles = nn.Parameter(torch.tensor(angles, dtype=dtype))
        generator = torch.Generator().manual_seed(seed)
        self.readout = _linear(self.output_dim, 1, generator, dtype)
        index = torch.arange(2**qubits)
        bits = (index[:, None] >> torch.arange(qubits - 1, -1, -1)) & 1
        signs = (1 - 2 * bits).to(dtype)
        edges = sorted({tuple(sorted((j, (j + 1) % qubits))) for j in range(qubits)})
        parity = sum(bits[:, a] * bits[:, b] for a, b in edges)
        self.register_buffer("signs", signs)
        self.register_buffer("zz", signs * signs.roll(-1, dims=1))
        self.register_buffer("phase", (1 - 2 * (parity % 2)).to(dtype))

    def _local_unitary(self, angles: Tensor, layer: int, j: int):
        c, s = torch.cos(angles[:, j] / 2), torch.sin(angles[:, j] / 2)
        phase = torch.exp(-0.5j * angles[:, (j + 1) % self.qubits])
        cb = torch.cos(self.angles[layer, j, 0] / 2)
        sb = torch.sin(self.angles[layer, j, 0] / 2)
        fixed = torch.exp(-0.5j * self.angles[layer, j, 1])
        return (
            fixed * (cb * phase * c - sb * phase.conj() * s),
            fixed * (-cb * phase * s - sb * phase.conj() * c),
            fixed.conj() * (sb * phase * c + cb * phase.conj() * s),
            fixed.conj() * (-sb * phase * s + cb * phase.conj() * c),
        )

    def statevectors(self, z: Tensor) -> Tensor:
        """Return differentiable exact statevectors; primarily useful for audits."""
        z = _validate_input(z, self.qubits, self.angles)
        angles = 2 * torch.atan(z)
        dtype = torch.complex128 if z.dtype == torch.float64 else torch.complex64
        state = torch.zeros((len(z), 2**self.qubits), dtype=dtype, device=z.device)
        state[:, 0] = 1
        for layer in range(self.layers):
            for j in range(self.qubits):
                u = [v[:, None, None] for v in self._local_unitary(angles, layer, j)]
                view = state.reshape(len(z), 2**j, 2, 2 ** (self.qubits - j - 1))
                a, b = view[:, :, 0], view[:, :, 1]
                state = torch.stack((u[0] * a + u[1] * b, u[2] * a + u[3] * b), dim=2)
                state = state.reshape(len(z), 2**self.qubits)
            if self.entangled:
                state = state * self.phase
        return state

    def _product_features(self, z: Tensor) -> Tensor:
        """Exact no-entanglement evaluation in O(batch * layers * qubits)."""
        angles = 2 * torch.atan(z)
        dtype = torch.complex128 if z.dtype == torch.float64 else torch.complex64
        a = torch.ones(z.shape, dtype=dtype, device=z.device)
        b = torch.zeros_like(a)
        for layer in range(self.layers):
            next_a, next_b = [], []
            for j in range(self.qubits):
                u = self._local_unitary(angles, layer, j)
                next_a.append(u[0] * a[:, j] + u[1] * b[:, j])
                next_b.append(u[2] * a[:, j] + u[3] * b[:, j])
            a, b = torch.stack(next_a, dim=1), torch.stack(next_b, dim=1)
        cross = a.conj() * b
        zmean = a.abs().square() - b.abs().square()
        return torch.cat((2 * cross.real, 2 * cross.imag, zmean,
                          zmean * zmean.roll(-1, dims=1)), dim=1)

    def embed(self, z: Tensor) -> Tensor:
        z = _validate_input(z, self.qubits, self.angles)
        if not self.entangled:
            return self._product_features(z)
        state = self.statevectors(z)
        probability = state.real.square() + state.imag.square()
        x, y = [], []
        for j in range(self.qubits):
            view = state.reshape(len(z), 2**j, 2, 2 ** (self.qubits - j - 1))
            cross = (view[:, :, 0].conj() * view[:, :, 1]).sum(dim=(1, 2))
            x.append(2 * cross.real)
            y.append(2 * cross.imag)
        return torch.cat((torch.stack(x, dim=1), torch.stack(y, dim=1),
                          probability @ self.signs, probability @ self.zz), dim=1)

    def forward(self, z: Tensor) -> Tensor:
        return self.readout(self.embed(z)).squeeze(-1)


class ClassicalMLPEnergy(nn.Module):
    """Same-input classical control with explicit features and scalar readout.

    Defaults to 8 -> 32 -> 32 -> 1 with SiLU hidden activations. ``hidden=6,
    depth=1`` gives the 61-parameter 8 -> 6 -> 1 capacity control; its feature
    count is six, so it is separate from the 32-feature head-matched control.
    """

    def __init__(self, input_dim=8, hidden=32, depth=2, seed=17, dtype=torch.float64):
        super().__init__()
        if any(not isinstance(v, int) or v < 1 for v in (input_dim, hidden, depth)):
            raise ValueError("MLP dimensions and depth must be positive integers")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Use float32 or float64 parameters")
        self.input_dim, self.output_dim = input_dim, hidden
        generator = torch.Generator().manual_seed(seed)
        layers = []
        for i in range(depth):
            layers.extend((_linear(input_dim if i == 0 else hidden, hidden, generator, dtype),
                           nn.SiLU()))
        self.encoder = nn.Sequential(*layers)
        self.readout = _linear(hidden, 1, generator, dtype)

    def embed(self, z: Tensor) -> Tensor:
        return self.encoder(_validate_input(z, self.input_dim, self.readout.weight))

    def forward(self, z: Tensor) -> Tensor:
        return self.readout(self.embed(z)).squeeze(-1)
