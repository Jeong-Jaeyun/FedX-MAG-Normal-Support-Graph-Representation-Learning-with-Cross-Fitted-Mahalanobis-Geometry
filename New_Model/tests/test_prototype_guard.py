import numpy as np
import pytest
import torch

from fedxmag_next.prototype_guard import VerifiedPrototypeMemory, fit_radii, keyed_distances
from fedxmag_next.window_bridge import replay_windows


def test_keyed_neighbors_use_equality_and_unknown_pairs_are_not_declared_normal():
    r = {"raw": torch.tensor([[0.0, 0.0], [10.0, 0.0]]), "deep": torch.tensor([[0.0], [10.0]])}
    q = {"raw": torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]), "deep": torch.ones(3, 1)}
    rk = {
        "source": np.array([2**64 - 1, 7], dtype=np.uint64),
        "destination": np.array([3, 3], dtype=np.uint64),
    }
    qk = {
        "source": np.array([2**64 - 1, 7, 2**64 - 1], dtype=np.uint64),
        "destination": np.array([3, 3, 9], dtype=np.uint64),
    }
    source = keyed_distances(r, q, rk, qk, "source")
    peer = keyed_distances(r, q, rk, qk, "peer")
    for name in r:
        assert source[name].tolist() == [1.0, 9.0, 1.0]
        assert peer[name][:2].tolist() == [1.0, 9.0]
        assert torch.isinf(peer[name][2])


def test_normal_memory_admits_only_after_feedback_resumes_and_obeys_fifo(tmp_path):
    values = {"raw": torch.tensor([[0.0], [1.0]]), "deep": torch.tensor([[0.0], [1.0]])}
    keys = {
        "source": np.array([2**64 - 1] * 2, dtype=np.uint64),
        "destination": np.array([3, 3], dtype=np.uint64),
    }
    radii = {scope: {name: 1.0 for name in values} for scope in ("source", "peer")}
    memory = VerifiedPrototypeMemory(values, keys, np.array([0, 1]), radii, maximum=3)
    current = {name: torch.tensor([[9.0], [10.0]]) for name in values}
    first = memory.query(60, current, keys, np.array([10, 20]))
    assert first["source"]["raw"].tolist() == [8.0, 9.0]
    with pytest.raises(ValueError, match="feedback"):
        memory.query(61, current, keys, np.array([60, 60]))
    with pytest.raises(ValueError, match="committed"):
        memory.state_dict()
    memory.feedback(verified_normal=True)
    assert memory.times.tolist() == [1, 10, 20]
    path = tmp_path / "normal.pt"
    torch.save(memory.state_dict(), path)
    restored = VerifiedPrototypeMemory.from_state_dict(torch.load(path, weights_only=True))
    second = memory.query(120, current, keys, np.array([70, 80]))
    replay = restored.query(120, current, keys, np.array([70, 80]))
    for scope in second:
        for name in second[scope]:
            torch.testing.assert_close(second[scope][name], replay[scope][name], rtol=0, atol=0)
            assert second[scope][name].tolist() == [0.0, 0.0]
    memory.feedback(verified_normal=False)
    restored.feedback(verified_normal=True)
    assert memory.times.tolist() == [1, 10, 20]
    assert restored.times.tolist() == [20, 70, 80]
    assert memory.maximum_seen == restored.maximum_seen == 3


def test_source_radius_excludes_unseen_entity_distances_and_needs_normal_support():
    reference = {"raw": torch.zeros(1, 1), "deep": torch.zeros(1, 1)}
    selection = {name: torch.arange(40.0).reshape(-1, 1) for name in reference}
    rk = {"source": np.ones(1, dtype=np.uint64), "destination": np.ones(1, dtype=np.uint64)}
    sk = {
        "source": np.r_[np.ones(32), np.zeros(8)].astype(np.uint64),
        "destination": np.ones(40, dtype=np.uint64),
    }
    radii, counts = fit_radii(reference, selection, rk, sk)
    assert radii["source"]["raw"] == pytest.approx(29.45)
    assert counts["peer"]["deep"] == 32
    sk["source"][:] = 0
    with pytest.raises(ValueError, match="Too few"):
        fit_radii(reference, selection, rk, sk)


def test_frozen_normal_feedback_reference_does_not_reinsert_augmented_energy():
    energy, labels = np.array([100.0, 80.0]), np.array([0, 1])
    arguments = (energy, np.ones(2, dtype=bool), labels, np.zeros(2, dtype=int), list(range(19)))
    ordinary = replay_windows(*arguments, capacity=19, policy="continuous")[0]
    fixed = replay_windows(
        *arguments, capacity=19, policy="continuous", feedback_energy=np.array([0.0, 0.0])
    )[0]
    assert ordinary.tolist() == [True, False]
    assert fixed.tolist() == [True, True]
    with pytest.raises(ValueError, match="aligned"):
        replay_windows(*arguments, capacity=19, feedback_energy=np.zeros(1))


def test_prototype_temporal_guard_blocks_familiar_carriers_without_using_query_labels(tmp_path):
    from fedxmag_next.prototype_temporal import replay_path

    values, reference = np.array([20.0, 0.0, 1.0]), np.arange(10.0)
    keys = {"source": np.ones(3, dtype=np.uint64), "destination": np.ones(3, dtype=np.uint64)}
    args = (
        values,
        reference,
        np.array([2.0, 0.0, 2.0]),
        np.arange(3),
        {i: np.array([i]) for i in range(3)},
        keys,
        np.array([10, 70, 130]),
        "top3",
        np.full(3, 10.0),
    )
    raw_risk, _ = replay_path(*args, "risk_fixed", "source", tmp_path / "risk.pt")
    guarded, info = replay_path(*args, "prototype_risk_fixed", "source", tmp_path / "joint.pt")
    assert raw_risk[1] > 10 and guarded[1] == 0
    assert guarded[0] == 20 and guarded[2] > 10
    assert info["checkpoint_resumed_exact"]
    values[0] = 100.0
    hard, _ = replay_path(*args, "prototype", "source", tmp_path / "hard.pt")
    continuous, _ = replay_path(
        *args, "prototype", "source", tmp_path / "continuous.pt", continuous_filter=True
    )
    assert hard[0] == 100 and 10 < continuous[0] < 100


def test_service_keys_distinguish_protocol_port_and_availability_without_changing_entity():
    from fedxmag_next.service_risk_keys import service_keys

    keys = {
        "source": np.full(4, 2**64 - 1, dtype=np.uint64),
        "destination": np.full(4, 7, dtype=np.uint64),
        "port": np.array([80, 80, 80, 80]),
    }
    protocol = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
    source = service_keys(keys, protocol, np.array([1, 1, 1, 0]), "source")
    peer = service_keys(keys, protocol, np.array([1, 1, 1, 0]), "peer")
    assert source[0] == source[2] and len(set(source)) == 3
    assert peer[0][:2] == (2**64 - 1, 7) and peer[0] == peer[2]


def test_service_refinement_stops_cross_service_carry_and_keeps_same_service_evidence(tmp_path):
    from fedxmag_next.prototype_temporal import replay_path

    values, reference = np.array([20.0, 1.0, 1.0]), np.arange(10.0)
    keys = {"source": np.ones(3, dtype=np.uint64), "destination": np.ones(3, dtype=np.uint64)}
    args = (
        values,
        reference,
        np.full(3, 2.0),
        np.arange(3),
        {i: np.array([i]) for i in range(3)},
        keys,
        np.array([10, 70, 130]),
        "top3",
        np.full(3, 10.0),
    )
    plain, _ = replay_path(
        *args, "prototype_risk_fixed", "source", tmp_path / "plain.pt", continuous_filter=True
    )
    refined, _ = replay_path(
        *args,
        "prototype_risk_fixed",
        "source",
        tmp_path / "service.pt",
        continuous_filter=True,
        risk_key_values=[(1, 80, 0, 1), (1, 443, 0, 1), (1, 80, 0, 1)],
    )
    assert plain[1] > 10 and refined[1] == 1
    assert refined[0] == plain[0] and refined[2] == plain[2]
    assert np.all(refined <= plain)
