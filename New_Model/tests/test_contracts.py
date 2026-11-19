from __future__ import annotations

import torch

from fedxmag_next.contracts import GraphWindow, collate, mask_features


def test_masking_never_targets_naturally_missing_values() -> None:
    item = GraphWindow(
        torch.ones(2, 2),
        torch.ones(1, 2),
        torch.tensor([[0], [1]], dtype=torch.long),
        torch.tensor([[True, False], [True, True]]),
        torch.tensor([[False, True]]),
        0,
    )
    masked = mask_features(collate([item]), 0.9, torch.Generator().manual_seed(8))
    assert not (masked.node_mask & ~item.node_available).any()
    assert not (masked.edge_mask & ~item.edge_available).any()
