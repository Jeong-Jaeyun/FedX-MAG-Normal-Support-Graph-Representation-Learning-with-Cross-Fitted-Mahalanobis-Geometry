from __future__ import annotations

import torch

from fedxmag_next.geometry import ConditionalMahalanobis


def test_conditional_support_preserves_separated_normal_modes() -> None:
    generator = torch.Generator().manual_seed(4)
    left = torch.randn(80, 2, generator=generator) * 0.15 + torch.tensor([-4.0, 0.0])
    right = torch.randn(80, 2, generator=generator) * 0.15 + torch.tensor([4.0, 0.0])
    support = ConditionalMahalanobis.fit(
        torch.cat((left, right)), max_components=2, min_component_size=32, seed=4
    )
    center_energy, modes = support.score(torch.tensor([[-4.0, 0.0], [4.0, 0.0]]))
    gap_energy, _ = support.score(torch.tensor([[0.0, 0.0]]))
    assert support.components == 2
    assert int(modes[0]) != int(modes[1])
    assert float(gap_energy[0]) > float(center_energy.max()) * 10.0


def test_component_energy_has_one_score_per_query_and_mode() -> None:
    reference = torch.randn(64, 3, generator=torch.Generator().manual_seed(3))
    support = ConditionalMahalanobis.fit(reference, max_components=2, min_component_size=20)
    score, mode = support.score(torch.randn(7, 3, generator=torch.Generator().manual_seed(5)))
    assert score.shape == (7,)
    assert mode.shape == (7,)
    assert torch.isfinite(score).all()
