import math

import torch

from fedxmag_next.conditional_support import HistoryConditionalSupport
from fedxmag_next.geometry import ConditionalMahalanobis


def test_conditional_mixture_matches_joint_density_divided_by_history_density():
    generator = torch.Generator().manual_seed(23)
    factors = torch.tril(torch.randn(3, 5, 5, generator=generator).double())
    factors.diagonal(dim1=1, dim2=2).copy_(torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0]))
    means = torch.randn(3, 5, generator=generator).double()
    weights = torch.tensor([0.2, 0.3, 0.5]).double()
    support = HistoryConditionalSupport(
        ConditionalMahalanobis(means, factors, weights.log(), torch.tensor([20, 30, 50])), 2
    )
    queries = torch.randn(19, 5, generator=generator).double()
    joint = torch.distributions.MultivariateNormal(means, scale_tril=factors)
    history = torch.distributions.MultivariateNormal(means[:, :2], scale_tril=factors[:, :2, :2])
    log_joint = torch.logsumexp(joint.log_prob(queries[:, None]) + weights.log(), dim=1)
    log_history = torch.logsumexp(history.log_prob(queries[:, None, :2]) + weights.log(), dim=1)
    expected = -2 * (log_joint - log_history) - 3 * math.log(2 * math.pi)
    actual = support.score(queries[:, 2:], queries[:, :2])
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    restored = HistoryConditionalSupport.from_state_dict(support.state_dict())
    torch.testing.assert_close(restored.score(queries[:, 2:], queries[:, :2]), actual)
    covariance = factors @ factors.transpose(1, 2)
    marginal = torch.distributions.MultivariateNormal(means[:, 2:], covariance[:, 2:, 2:])
    expected_marginal = -2 * torch.logsumexp(
        marginal.log_prob(queries[:, None, 2:]) + weights.log(), dim=1
    ) - 3 * math.log(2 * math.pi)
    torch.testing.assert_close(
        support.marginal_score(queries[:, 2:]), expected_marginal, atol=1e-10, rtol=1e-10
    )


def test_extreme_history_alone_does_not_raise_conditional_energy():
    # y = 3*h + residual. Joint surprise grows with h; conditional surprise
    # should be identical when the residual is identical, even for extreme h.
    factor = torch.tensor([[[2.0, 0.0], [6.0, 0.5]]], dtype=torch.float64)
    support = HistoryConditionalSupport(
        ConditionalMahalanobis(
            torch.zeros(1, 2).double(), factor, torch.zeros(1).double(), torch.tensor([100])
        ),
        1,
    )
    history = torch.tensor([[0.0], [10.0], [10000.0]], dtype=torch.float64)
    current = 3 * history + 0.25
    energy = support.score(current, history)
    torch.testing.assert_close(energy, energy[0].expand_as(energy), atol=1e-10, rtol=1e-10)
