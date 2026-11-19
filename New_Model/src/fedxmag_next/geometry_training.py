"""Jointly optimize normal Mahalanobis geometry while preserving all raw inputs.

Each training minibatch has disjoint normal anchor/probe rows. Matched-history
negatives supply repulsion; positive energy alone is never the objective. This
is row cross-fitting inside training, not a temporal exchangeability guarantee.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .context_association import ContextPairSampler
from .geometry import _shrinkage_covariance


class ResidualFeatureMap(nn.Module):
    def __init__(self, dimension, hidden):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

    def forward(self, values):
        return torch.cat((values, self.network(values)), dim=1)


class GeometryResidualEncoder(nn.Module):
    def __init__(self, dimension, hidden=64):
        super().__init__()
        self.input_dimension = dimension
        # Same layers[:-1] interface; no unused classification head.
        self.layers = nn.Sequential(ResidualFeatureMap(dimension, hidden), nn.Identity())

    def forward(self, values):
        return self.layers(values)


def geometry_loss(anchor, positive, negative, raw_dimension):
    if len(anchor) < 2 or len(positive) != len(negative):
        raise ValueError("Independent normal anchors and aligned contrast pairs required")
    tail = anchor[:, raw_dimension:]
    center = tail.mean(0)
    scale = tail.std(0, unbiased=False).clamp_min(1e-6)

    def normalized(values):
        return torch.cat(
            (values[:, :raw_dimension], (values[:, raw_dimension:] - center) / scale), 1
        )

    reference = normalized(anchor)
    mean, covariance = _shrinkage_covariance(reference, 1e-5)
    factor = torch.linalg.cholesky(covariance)
    query = normalized(torch.cat((positive, negative))).double()
    white = torch.linalg.solve_triangular(factor, (query - mean).T, upper=False)
    energy = white.square().sum(0) / query.shape[1]
    real, changed = energy[: len(positive)], energy[len(positive) :]
    ranking = F.softplus(1 + real - changed).mean()
    compact = real.mean()
    variance = F.relu(0.1 - tail.std(0, unbiased=False)).mean()
    loss = ranking + 0.1 * compact + 0.1 * variance
    return loss, real.mean(), changed.mean()


def train_geometry(
    model,
    reference,
    selection,
    reference_cohorts,
    selection_cohorts,
    context_columns,
    *,
    seed,
    epochs,
):
    sampler = ContextPairSampler(reference, reference_cohorts, context_columns)
    validation_sampler = ContextPairSampler(selection, selection_cohorts, context_columns)
    if min(sampler.eligible, validation_sampler.eligible) < 64 or epochs < 1:
        raise ValueError("Insufficient changed-context pairs for geometry training")
    generator = torch.Generator().manual_seed(seed)
    vi, vd = validation_sampler.sample(torch.Generator().manual_seed(7189))
    validation_positive = selection[vi]
    validation_negative = validation_sampler.negative(vi, vd)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    history, best_state, best_value = [], None, float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        ids, donors = sampler.sample(generator)
        order = torch.randperm(len(ids), generator=generator)
        total, count = 0.0, 0
        overlap = 0
        for positions in order.split(256):
            if len(positions) < 64:
                continue
            split = len(positions) // 2
            anchors = ids[positions[:split]]
            probes = ids[positions[split:]]
            overlap += int(torch.isin(anchors, probes).sum())
            if overlap:
                raise ValueError("A geometry probe entered its own normal covariance")
            negatives = sampler.negative(probes, donors[positions[split:]])
            embedded = model(torch.cat((reference[anchors], reference[probes], negatives)))
            a, p, n = embedded.split((len(anchors), len(probes), len(probes)))
            loss, _, _ = geometry_loss(a, p, n, model.input_dimension)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite geometry objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(
                parameter.grad is None or not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise ValueError("Missing/nonfinite joint covariance gradient")
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(probes)
            count += len(probes)
        model.eval()
        with torch.no_grad():
            anchor = model(reference)
            positive = model(validation_positive)
            negative = model(validation_negative)
            loss, real, changed = geometry_loss(anchor, positive, negative, model.input_dimension)
            value = float(loss)
        entry = {
            "epoch": epoch,
            "train_geometry_loss": total / count,
            "selection_geometry_loss": value,
            "selection_normal_energy_per_dimension": float(real),
            "selection_changed_energy_per_dimension": float(changed),
            "anchor_probe_row_overlap": overlap,
            "train_changed_context_pairs": sampler.eligible,
        }
        history.append(entry)
        if value < best_value:
            best_value = value
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % 10 == 0:
            print(entry, flush=True)
    if best_state is None:
        raise ValueError("No finite joint-geometry checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return history
