import numpy as np
import torch

from fedxmag_next.evidence_experiment import evidence_path
from fedxmag_next.normal_views import CurrentNormalProjector


def test_current_view_is_invariant_to_past_context_and_uses_marginal_covariance():
    factor = torch.tensor([[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.0, 1.0, 3.0]], dtype=torch.float64)
    geometry = {
        "means": torch.zeros(1, 3, dtype=torch.float64),
        "cholesky": factor[None],
        "log_prior": torch.zeros(1),
        "counts": torch.tensor([100]),
    }
    state = {
        "format_version": 2,
        "feature_names": ["current_a", "context_old", "current_b"],
        "input_mean": torch.zeros(3),
        "input_scale": torch.ones(3),
        "input_keep": None,
        "seed_models": [],
        "raw_supports": {"raw_mahalanobis_m1": geometry},
    }
    projector = CurrentNormalProjector(state)
    values = torch.tensor([[3.0, 0.0, 4.0], [3.0, 100.0, 4.0]])
    result = projector.project(values)
    assert torch.equal(result["current"][0], result["current"][1])
    assert not torch.equal(result["raw"][0], result["raw"][1])
    covariance = factor @ factor.T
    selected = [0, 2]
    current_factor = projector.state["current_geometry"]["cholesky"][0]
    torch.testing.assert_close(current_factor @ current_factor.T, covariance[selected][:, selected])


def test_current_guard_blocks_carry_without_changing_the_local_filter(tmp_path):
    values, reference = np.array([4.0, 1.0]), np.array([1.0, 2.0])
    ratios = np.array([2.0, 2.0])
    keys = {
        "source": np.array([7, 7], dtype=np.uint64),
        "destination": np.array([8, 8], dtype=np.uint64),
    }
    args = (
        values,
        reference,
        ratios,
        np.array([0, 1]),
        {0: np.array([0]), 1: np.array([1])},
        keys,
        np.array([10, 70]),
        np.array([99.0, 99.0]),
        "source",
        "sum",
        "weak",
    )
    ordinary, _ = evidence_path(*args, tmp_path / "ordinary.pt", aggregation="top3")
    guarded, _ = evidence_path(
        *args, tmp_path / "guarded.pt", aggregation="top3", guard_ratios=np.array([2.0, 0.1])
    )
    assert ordinary[0] == guarded[0] == 4
    assert ordinary[1] > guarded[1] == 1
