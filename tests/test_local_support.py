import pytest
import torch

from fedxmag.local_support import LocalSupport


def test_local_novelty_matches_bruteforce_reachability() -> None:
    reference = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [5.0, 1.0]], dtype=torch.float64)
    query = torch.tensor([[0.5, 0.5], [20.0, 20.0]], dtype=torch.float64)
    model = LocalSupport.fit(reference, k=2)
    distance = ((reference[:, None] - reference[None, :]).square().sum(-1)).sqrt()
    distance.fill_diagonal_(torch.inf)
    d, idx = distance.sort(dim=1)
    radius = d[:, 1]
    reference_lrd = 1 / torch.maximum(d[:, :2], radius[idx[:, :2]]).mean(1)
    query_d = (query[:, None] - reference[None, :]).square().sum(-1).sqrt()
    qd, qi = query_d.sort(dim=1)
    lrd = 1 / torch.maximum(qd[:, :2], radius[qi[:, :2]]).mean(1)
    actual = model.score(query)
    torch.testing.assert_close(actual["knn"], qd[:, :2].mean(1))
    torch.testing.assert_close(actual["lof"], (reference_lrd[qi[:, :2]] / lrd[:, None]).mean(1))


def test_whitened_neighbours_and_empty_query() -> None:
    reference = torch.randn(40, 4, generator=torch.Generator().manual_seed(1)).double()
    query = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    model = LocalSupport.fit(reference, k=5, whiten=True)
    residual = query - reference
    inverse = torch.cholesky_inverse(model.factor)
    expected = ((residual @ inverse) * residual).sum(1).sqrt().sort().values[:5].mean()
    torch.testing.assert_close(model.score(query)["knn"][0], expected)
    assert model.score(query[:0])["knn"].numel() == 0
    with pytest.raises(ValueError):
        LocalSupport.fit(reference, k=40)


def test_duplicate_reference_stays_finite_and_queries_do_not_change_reference() -> None:
    reference = torch.tensor([[0.0], [0.0], [0.0], [1.0], [2.0]], dtype=torch.float64)
    model = LocalSupport.fit(reference, k=2)
    query = torch.tensor([[0.0], [100.0]], dtype=torch.float64)
    score = model.score(query)
    assert torch.isfinite(score["lof"]).all()
    assert score["knn"][1] > score["knn"][0]
    torch.testing.assert_close(model.score(query[:1])["lof"], score["lof"][:1])
