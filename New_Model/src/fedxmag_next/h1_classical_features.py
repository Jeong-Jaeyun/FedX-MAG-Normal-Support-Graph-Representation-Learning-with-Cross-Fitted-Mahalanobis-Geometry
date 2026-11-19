"""Normal-fitted PCA and explicit classical feature controls for H1."""

import numpy as np
import torch


def _values(x):
    x = torch.as_tensor(x).detach().cpu().double()
    if x.ndim != 2 or not torch.isfinite(x).all():
        raise ValueError("Finite matrix required")
    return x


class SourcePCA:
    def __init__(self, state):
        self.state = state

    @classmethod
    def fit(cls, reference, dimension=8):
        x = _values(reference)
        if len(x) < 2 or not 1 <= dimension <= x.shape[1]:
            raise ValueError("Invalid PCA fit dimensions")
        mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(0.05)
        v = (x - mean) / scale
        eigen, vectors = torch.linalg.eigh(v.T @ v / len(v))
        axes = vectors[:, -dimension:].flip(1)
        signs = axes[axes.abs().argmax(0), torch.arange(dimension)].sign()
        axes = axes * signs
        latent = v @ axes
        return cls(
            {
                "mean": mean,
                "scale": scale,
                "axes": axes,
                "latent_mean": latent.mean(0),
                "latent_scale": latent.std(0, unbiased=False).clamp_min(1e-6),
                "explained_variance_ratio": eigen[-dimension:].flip(0)
                / eigen.sum().clamp_min(1e-12),
            }
        )

    def normalize(self, x):
        x = _values(x)
        if x.shape[1] != len(self.state["mean"]):
            raise ValueError("Input schema dimension differs")
        return (x - self.state["mean"]) / self.state["scale"]

    def transform(self, x):
        return (self.normalize(x) @ self.state["axes"] - self.state["latent_mean"]) / self.state[
            "latent_scale"
        ]

    def state_dict(self):
        return self.state

    @classmethod
    def from_state_dict(cls, state):
        return cls(state)


class ClassicalFeatureMap:
    def __init__(self, state):
        self.state = state

    @classmethod
    def fit(cls, kind, z_reference, seed=17, output_dimension=32, landmarks=128):
        z = _values(z_reference)
        rng = np.random.default_rng(seed)
        d = z.shape[1]
        state = {"kind": kind, "dimension": d}

        def normal(shape):
            return torch.from_numpy(rng.normal(size=shape))

        if kind == "rff":
            distances = torch.pdist(z[:512]).square()
            gamma = 1 / float(distances.median().clamp_min(1e-6))
            state.update(
                w=normal((d, output_dimension)) * np.sqrt(2 * gamma),
                b=torch.from_numpy(rng.uniform(0, 2 * np.pi, output_dimension)),
            )
        elif kind == "random_mlp":
            state.update(
                w=normal((d, output_dimension)) / np.sqrt(d),
                b=normal((output_dimension,)),
                w2=normal((output_dimension, output_dimension)) / np.sqrt(output_dimension),
            )
        elif kind in ("rbf_nystrom", "polynomial_nystrom"):
            state["landmarks"] = z[rng.permutation(len(z))[:landmarks]].clone()
            state["gamma"] = 1 / float(torch.pdist(z[:512]).square().median().clamp_min(1e-6))
            obj = cls(state)
            k = obj._kernel(state["landmarks"])
            eigen, vectors = torch.linalg.eigh(k)
            floor = max(float(eigen.max()) * 1e-8, 1e-10)
            state["inverse_root"] = vectors / eigen.clamp_min(floor).sqrt()[None, :]
        elif kind not in ("linear", "trigonometric"):
            raise ValueError("Unknown classical map")
        return cls(state)

    def _kernel(self, z):
        if self.state["kind"] == "rbf_nystrom":
            return torch.exp(
                -self.state["gamma"] * torch.cdist(z, self.state["landmarks"]).square()
            )
        return (1 + z @ self.state["landmarks"].T / self.state["dimension"]) ** 3

    def transform(self, z):
        z = _values(z)
        s, kind = self.state, self.state["kind"]
        if z.shape[1] != s["dimension"]:
            raise ValueError("Latent dimension differs")
        if kind == "linear":
            return z.clone()
        if kind == "trigonometric":
            return torch.cat((z.sin(), z.cos(), (2 * z).sin(), (2 * z).cos()), 1)
        if kind == "rff":
            return (z @ s["w"] + s["b"]).cos() * np.sqrt(2 / len(s["b"]))
        if kind == "random_mlp":
            return torch.nn.functional.silu(z @ s["w"] + s["b"]) @ s["w2"]
        return self._kernel(z) @ s["inverse_root"]

    def state_dict(self):
        return self.state

    @classmethod
    def from_state_dict(cls, state):
        return cls(state)
