"""Frozen eight-minute quantum/classical predictors; no target fitting."""

import numpy as np
import torch

from .quantum_features import ProjectedQuantumMap

METHODS = (
    "ar_ridge",
    "delay_support",
    "esn",
    "gru",
    "quantum",
    "quantum_noent",
    "quantum_zero_history",
    "quantum_shuffled",
)
RIDGES = (1e-4, 1e-2, 1.0, 100.0)


class SequentialMap:
    def __init__(self, kind, seed=17, state=None):
        if kind not in METHODS or kind == "gru":
            raise ValueError("Unknown fixed temporal map")
        self.kind, self.seed = kind, seed
        rng = np.random.default_rng(seed)
        self.circuit = ProjectedQuantumMap(
            qubits=8, layers=1, seed=seed, entangled=kind != "quantum_noent", chunk_size=256
        )
        self.permutation = rng.permutation(8)
        w = rng.normal(size=(32, 32))
        self.esn_w = w * (0.9 / np.max(np.abs(np.linalg.eigvals(w))))
        self.esn_in = rng.normal(size=(9, 32)) / 3
        self.esn_bias = rng.normal(size=32) * 0.1
        if state is not None:
            self.circuit = ProjectedQuantumMap.from_state_dict(state["circuit"])
            for name in ("permutation", "esn_w", "esn_in", "esn_bias"):
                setattr(self, name, state[name].numpy().copy())

    def state_dict(self):
        return dict(
            kind=self.kind,
            seed=self.seed,
            circuit=self.circuit.state_dict(),
            **{
                name: torch.from_numpy(getattr(self, name).copy())
                for name in ("permutation", "esn_w", "esn_in", "esn_bias")
            },
        )

    def transform(self, x, mask):
        x, mask = np.asarray(x, dtype=np.float64), np.asarray(mask, dtype=np.float64)
        if (
            x.shape[1:] != (8, 8)
            or mask.shape != x.shape[:2]
            or not np.isfinite(x).all()
            or not np.isin(mask, (0, 1)).all()
        ):
            raise ValueError("Finite [batch,8,8] history and [batch,8] mask required")
        x = x * mask[:, :, None]
        if self.kind in ("ar_ridge", "delay_support"):
            return np.c_[x.reshape(len(x), -1), mask]
        if self.kind == "esn":
            hidden = np.zeros((len(x), 32))
            for j in range(8):
                hidden = np.tanh(
                    np.c_[x[:, j], mask[:, j]] @ self.esn_in + hidden @ self.esn_w + self.esn_bias
                )
            return np.c_[hidden, mask]
        if self.kind == "quantum_zero_history":
            x, mask = np.zeros_like(x), np.zeros_like(mask)
        elif self.kind == "quantum_shuffled":
            # Canonical content ordering removes chronology before applying the
            # seeded permutation. Reversing the original history gives the same
            # representation, unlike one fixed invertible time permutation.
            canonical = np.empty(x.shape[:2], dtype=np.int64)
            for i in range(len(x)):
                keys = np.c_[mask[i], x[i]]
                canonical[i] = np.lexsort(keys[:, ::-1].T)
            canonical = canonical[:, self.permutation]
            x = np.take_along_axis(x, canonical[:, :, None], axis=1)
            mask = np.take_along_axis(mask, canonical, axis=1)
        result = np.empty((len(x), 32))
        for start in range(0, len(x), 256):
            xx, mm = x[start : start + 256], mask[start : start + 256]
            state = np.zeros((len(xx), 256), dtype=np.complex128)
            state[:, 0] = 1
            for t in range(8):
                # Missing slots have a distinct rotation and an explicit readout mask.
                angles = 2 * np.arctan(xx[:, t]) + (1 - mm[:, t, None]) * (np.pi / 2)
                for j in range(8):
                    u = [v[:, None, None] for v in self.circuit._local_unitary(angles, 0, j)]
                    view = state.reshape(len(state), 2**j, 2, 2 ** (7 - j))
                    a, b = view[:, :, 0].copy(), view[:, :, 1].copy()
                    view[:, :, 0], view[:, :, 1] = u[0] * a + u[1] * b, u[2] * a + u[3] * b
                if self.circuit.entangled:
                    state *= self.circuit.phase
            result[start : start + len(xx)] = self.circuit._expectations(state)
        return np.c_[result, mask]


def ridge_fit(x, y, selected_x, selected_y):
    """Choose only normal-validation MSE, without refitting on validation observations."""
    x, y, sx, sy = [torch.as_tensor(v, dtype=torch.float64) for v in (x, y, selected_x, selected_y)]
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(0.05)
    xx, ss = (x - mean) / scale, (sx - mean) / scale
    center = y.mean(0)
    eye = torch.eye(xx.shape[1], dtype=torch.float64)
    candidates = []
    for strength in RIDGES:
        weight = torch.linalg.solve(
            xx.T @ xx / len(xx) + strength * eye, xx.T @ (y - center) / len(xx)
        )
        mse = float(((ss @ weight + center - sy) ** 2).mean())
        candidates.append((mse, strength, weight))
    mse, strength, weight = min(candidates, key=lambda v: v[0])
    return dict(
        mean=mean, scale=scale, center=center, weight=weight, ridge=strength, validation_mse=mse
    )


def ridge_predict(state, x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return (((x - state["mean"]) / state["scale"]) @ state["weight"] + state["center"]).numpy()


class SmallGRU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = torch.nn.GRU(9, 32, batch_first=True)
        self.head = torch.nn.Linear(32, 8)

    def forward(self, x):
        _, h = self.cell(x)
        return self.head(h[0])


def gru_inputs(x, mask):
    # Bounded inputs stabilize training; applied to every source/target split identically.
    return torch.as_tensor(
        np.concatenate((2 * np.arctan(x * mask[:, :, None]) / np.pi, mask[:, :, None]), axis=2),
        dtype=torch.float32,
    )


def fit_gru(x, mask, y, sx, smask, sy, seed, epochs=100):
    torch.manual_seed(seed)
    model = SmallGRU()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0001)
    tx, vx = gru_inputs(x, mask), gru_inputs(sx, smask)
    target, valid = (
        torch.as_tensor(y, dtype=torch.float32),
        torch.as_tensor(sy, dtype=torch.float32),
    )
    best_loss, best, best_epoch = float("inf"), None, 0
    gen = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        for ids in torch.randperm(len(tx), generator=gen).split(128):
            optimizer.zero_grad()
            loss = (model(tx[ids]) - target[ids]).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = float((model(vx) - valid).square().mean())
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch + 1
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return dict(weights=best, validation_mse=best_loss, selected_epoch=best_epoch, epochs=epochs)


def predict_gru(state, x, mask):
    model = SmallGRU()
    model.load_state_dict(state["weights"])
    model.eval()
    with torch.no_grad():
        return torch.cat([model(v) for v in gru_inputs(x, mask).split(512)]).double().numpy()
