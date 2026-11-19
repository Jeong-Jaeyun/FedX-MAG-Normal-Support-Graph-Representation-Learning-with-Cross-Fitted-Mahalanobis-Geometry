import numpy as np
import pytest
import torch

from fedxmag_next.rank_consensus import RankConsensus


def test_consensus_distinguishes_isolated_model_excursion_from_shared_anomaly():
    calibration = np.column_stack([np.arange(39)] * 3)
    isolated, shared = [100, 10, 10], [100, 100, 100]
    for rule in ("any", "mean", "median", "all"):
        model = RankConsensus(calibration, 39, rule)
        flag, valid, _ = model.query(100, isolated)
        assert valid
        if rule == "any":
            assert flag
        elif rule in ("median", "all"):
            assert not flag
        model.feedback(verified_normal=False)
        assert model.query(101, shared)[0]


def test_consensus_feedback_is_causal_and_disk_restore_preserves_each_query(tmp_path):
    random = np.random.default_rng(920)
    model = RankConsensus(random.normal(size=(50, 3)), 39, "median")
    for index, query in enumerate(random.normal(size=(80, 3))):
        minute = 1000 + 2 * index
        before = np.asarray(model.history).copy()
        model.query(minute, query)
        with pytest.raises(ValueError, match="feedback"):
            model.state_dict()
        with pytest.raises(ValueError, match="feedback"):
            model.query(minute + 1, query)
        verified = minute % 3 == 0
        model.feedback(verified_normal=verified, scores=query * 0.5)
        if verified:
            assert np.array_equal(model.history[-1], query * 0.5)
        else:
            assert np.array_equal(model.history, before)
        path = tmp_path / "consensus.pt"
        torch.save(model.state_dict(), path)
        restored = RankConsensus.from_state_dict(torch.load(path, weights_only=True))
        future = random.normal(size=3)
        assert model.query(minute + 1, future) == restored.query(minute + 1, future)
        model.feedback(verified_normal=False)
        restored.feedback(verified_normal=False)
        assert model.state_dict() == restored.state_dict()
