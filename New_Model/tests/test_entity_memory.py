import torch

from fedxmag_next.normal_memory import entity_distances


def test_memory_cannot_borrow_a_different_entity_and_keys_only_define_equality():
    reference = torch.tensor([[0.0, 0.0], [3.0, 4.0], [10.0, 10.0]])
    query = torch.tensor([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]])
    reference_keys = torch.tensor([10, 20, 10])
    query_keys = torch.tensor([20, 30, 10])
    distances = entity_distances(reference, query, reference_keys, query_keys)
    torch.testing.assert_close(distances, torch.tensor([5.0, torch.inf, 0.0]).double())
    relabeled = entity_distances(reference, query, reference_keys * 37 + 11, query_keys * 37 + 11)
    torch.testing.assert_close(distances, relabeled)
