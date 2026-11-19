"""Fixed projected circuit, exact CPU simulation; not a quantum speedup."""

import numpy as np
import torch


class ProjectedQuantumMap:
    def __init__(self, qubits=8, layers=2, seed=17, entangled=True, chunk_size=1024):
        if not 2 <= qubits <= 12 or layers < 1 or chunk_size < 1:
            raise ValueError("Invalid circuit size")
        self.qubits, self.layers, self.seed = qubits, layers, seed
        self.entangled, self.chunk_size = entangled, chunk_size
        self.angles = np.random.default_rng(seed).uniform(-np.pi, np.pi, (layers, qubits, 2))
        index = np.arange(2**qubits)
        bits = (index[:, None] >> np.arange(qubits - 1, -1, -1)) & 1
        self.signs = 1 - 2 * bits
        self.zz = self.signs * np.roll(self.signs, -1, axis=1)
        edges = {tuple(sorted((j, (j + 1) % qubits))) for j in range(qubits)}
        parity = sum(bits[:, a] * bits[:, b] for a, b in edges)
        self.phase = 1 - 2 * (parity % 2)

    def state_dict(self):
        return {
            "format_version": 1,
            "qubits": self.qubits,
            "layers": self.layers,
            "seed": self.seed,
            "entangled": self.entangled,
            "angles": torch.from_numpy(self.angles.copy()),
        }

    @classmethod
    def from_state_dict(cls, state):
        if state["format_version"] != 1:
            raise ValueError("Unsupported circuit format")
        obj = cls(**{k: state[k] for k in ("qubits", "layers", "seed", "entangled")})
        obj.angles = state["angles"].numpy().copy()
        return obj

    def _gate(self, state, j, angle, kind):
        view = state.reshape(len(state), 2**j, 2, 2 ** (self.qubits - j - 1))
        angle = np.asarray(angle)
        if angle.ndim:
            angle = angle[:, None, None]
        if kind == "y":
            c, s = np.cos(angle / 2), np.sin(angle / 2)
            a, b = view[:, :, 0].copy(), view[:, :, 1].copy()
            view[:, :, 0] = c * a - s * b
            view[:, :, 1] = s * a + c * b
        elif kind == "z":
            view[:, :, 0] *= np.exp(-0.5j * angle)
            view[:, :, 1] *= np.exp(0.5j * angle)
        else:
            raise ValueError("Unknown rotation")

    def _local_unitary(self, angles, layer, j):
        c, s = np.cos(angles[:, j] / 2), np.sin(angles[:, j] / 2)
        phase = np.exp(-0.5j * angles[:, (j + 1) % self.qubits])
        cb, sb = np.cos(self.angles[layer, j, 0] / 2), np.sin(self.angles[layer, j, 0] / 2)
        fixed = np.exp(-0.5j * self.angles[layer, j, 1])
        return (
            fixed * (cb * phase * c - sb * phase.conj() * s),
            fixed * (-cb * phase * s - sb * phase.conj() * c),
            fixed.conjugate() * (sb * phase * c + cb * phase.conj() * s),
            fixed.conjugate() * (-sb * phase * s + cb * phase.conj() * c),
        )

    def _states(self, z):
        a = 2 * np.arctan(z)
        state = np.zeros((len(z), 2**self.qubits), dtype=np.complex128)
        state[:, 0] = 1
        for layer in range(self.layers):
            for j in range(self.qubits):
                u = [v[:, None, None] for v in self._local_unitary(a, layer, j)]
                view = state.reshape(len(state), 2**j, 2, 2 ** (self.qubits - j - 1))
                left, right = view[:, :, 0].copy(), view[:, :, 1].copy()
                view[:, :, 0] = u[0] * left + u[1] * right
                view[:, :, 1] = u[2] * left + u[3] * right
            if self.entangled:
                state *= self.phase
        return state

    def _input(self, z):
        z = torch.as_tensor(z).detach().cpu().double().numpy()
        if z.ndim != 2 or z.shape[1] != self.qubits or not np.isfinite(z).all():
            raise ValueError("Circuit inputs must be finite [rows,qubits]")
        return z

    def statevectors(self, z):
        z = self._input(z)
        return np.concatenate(
            [self._states(z[i : i + self.chunk_size]) for i in range(0, len(z), self.chunk_size)]
        )

    def _expectations(self, state):
        prob = np.abs(state) ** 2
        x, y = [], []
        for j in range(self.qubits):
            v = state.reshape(len(state), 2**j, 2, 2 ** (self.qubits - j - 1))
            cross = (v[:, :, 0].conj() * v[:, :, 1]).sum(axis=(1, 2))
            x.append(2 * cross.real)
            y.append(2 * cross.imag)
        return np.concatenate(
            (np.stack(x, 1), np.stack(y, 1), prob @ self.signs, prob @ self.zz), axis=1
        )

    def transform(self, z):
        z = self._input(z)
        if not self.entangled:
            # Exact product-state evaluation; independent dense oracle tests cover it.
            angles = 2 * np.arctan(z)
            a = np.ones(z.shape, dtype=np.complex128)
            b = np.zeros(z.shape, dtype=np.complex128)
            for layer in range(self.layers):
                for j in range(self.qubits):
                    u = self._local_unitary(angles, layer, j)
                    left, right = a[:, j].copy(), b[:, j].copy()
                    a[:, j], b[:, j] = u[0] * left + u[1] * right, u[2] * left + u[3] * right
            cross, zz = a.conj() * b, np.abs(a) ** 2 - np.abs(b) ** 2
            return torch.from_numpy(
                np.concatenate(
                    (2 * cross.real, 2 * cross.imag, zz, zz * np.roll(zz, -1, axis=1)), axis=1
                )
            )
        result = np.empty((len(z), 4 * self.qubits))
        for i in range(0, len(z), self.chunk_size):
            result[i : i + self.chunk_size] = self._expectations(
                self._states(z[i : i + self.chunk_size])
            )
        return torch.from_numpy(result)

    def transform_shots(self, z, shots=1024, seed=731):
        """Three measurement settings, shared Z shots for Z and ZZ correlations."""
        if shots < 1:
            raise ValueError("Positive shot count required")
        z = self._input(z)
        rng = np.random.default_rng(seed)
        out = np.empty((len(z), 4 * self.qubits))
        for start in range(0, len(z), self.chunk_size):
            states = self._states(z[start : start + self.chunk_size])
            for basis, offset in (("x", 0), ("y", self.qubits), ("z", 2 * self.qubits)):
                rotated = states.copy()
                for j in range(self.qubits):
                    if basis == "y":
                        self._gate(rotated, j, -np.pi / 2, "z")
                    if basis != "z":
                        self._gate(rotated, j, -np.pi / 2, "y")
                for k, state in enumerate(rotated):
                    p = np.abs(state) ** 2
                    counts = rng.multinomial(shots, p / p.sum()) / shots
                    out[start + k, offset : offset + self.qubits] = counts @ self.signs
                    if basis == "z":
                        out[start + k, 3 * self.qubits :] = counts @ self.zz
        return torch.from_numpy(out)
