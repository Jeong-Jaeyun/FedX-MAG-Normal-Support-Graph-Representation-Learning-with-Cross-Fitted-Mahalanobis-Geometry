import torch

from fedxmag_next.node_experiment import pool_nodes


def test_directed_pooling_preserves_both_directions_and_empty_incidence():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    outgoing = torch.tensor([0, 0, 1])
    incoming = torch.tensor([1, 1, 2])
    pooled = pool_nodes(values, outgoing, incoming, 3)
    torch.testing.assert_close(pooled[0, :2], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(pooled[0, 2:4], torch.tensor([3.0, 4.0]))
    assert torch.all(pooled[0, 5:] == 0)
    assert torch.all(pooled[2, :5] == 0)
    torch.testing.assert_close(pooled[1, 5:7], torch.tensor([2.0, 3.0]))
    assert torch.isfinite(pooled).all()
