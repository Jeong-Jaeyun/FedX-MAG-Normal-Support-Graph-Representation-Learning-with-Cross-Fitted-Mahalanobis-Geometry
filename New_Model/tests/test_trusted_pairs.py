from fedxmag_next.trusted_adjacency import TrustedPairs


def test_recent_graph_memory_expires_duplicate_counts_but_keeps_source_anchors():
    memory = TrustedPairs([(1, 2)], capacity=2)
    assert memory.score([(3, 4)])["new_pair_fraction"] == 1
    memory.update([(3, 4)])
    assert memory.score([(3, 4)])["new_endpoint_fraction"] == 0
    memory.update([(5, 6), (5, 6)])
    assert memory.score([(3, 4)])["new_endpoint_fraction"] == 1
    memory.update([(7, 8)])
    assert memory.score([(5, 6)])["new_pair_fraction"] == 0
    memory.update([(7, 8)])
    assert memory.score([(5, 6)])["new_endpoint_fraction"] == 1
    assert memory.score([(1, 2)])["new_pair_fraction"] == 0
