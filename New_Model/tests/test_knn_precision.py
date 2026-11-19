from __future__ import annotations

import pytest
import torch

from fedxmag_next.experiments import knn_scores


def test_near_duplicate_distances_are_not_lost_in_large_squared_norms() -> None:
    reference = torch.full((32, 73), 10000.0, dtype=torch.float64)
    query = reference[:2].clone()
    query[0, 0] += 0.125
    query[1, 0] += 0.25
    expected = torch.tensor([0.125, 0.25], dtype=torch.float64)
    torch.testing.assert_close(knn_scores(reference, query), expected, rtol=1e-6, atol=1e-6)


def test_empty_calibration_batch_is_supported_but_empty_reference_is_not() -> None:
    reference = torch.ones(32, 3)
    result = knn_scores(reference, torch.empty(0, 3))
    assert result.shape == (0,)
    assert result.dtype == torch.float64
    with pytest.raises(ValueError, match="reference cannot be empty"):
        knn_scores(torch.empty(0, 3), reference)
