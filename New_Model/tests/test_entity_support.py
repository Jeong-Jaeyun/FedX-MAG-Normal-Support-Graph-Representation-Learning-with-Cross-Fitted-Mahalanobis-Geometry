from argparse import Namespace

import numpy as np
import pytest
import torch

from fedxmag_next.entity_support import EntitySupportDetector, fit_entity_state, route_entities
from fedxmag_next.record_prequential import build_source


def test_routing_preserves_unsigned_key_equality_and_unseen_fallback():
    vocabulary = np.array([2, 2**63 + 5, 2**64 - 1], dtype=np.uint64)
    stored = torch.from_numpy(vocabulary.view(np.int64).copy())
    query = np.array([2**64 - 1, 5, 2, 2**63 + 5], dtype=np.uint64)
    assert route_entities(query, stored).tolist() == [2, 3, 0, 1]
    with pytest.raises(ValueError):
        route_entities(query.astype(np.float64), stored)


def test_entity_model_routes_normals_without_changing_encoder_and_restores(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(626)
    x = rng.normal(size=(256, 6)).astype(np.float32)
    ids = np.tile(np.repeat(np.array([2**63 + 5, 2**64 - 1], dtype=np.uint64), 64), 2)
    x[ids == 2**64 - 1] += 3
    data = {
        "features": x,
        "feature_names": np.array([f"f{i}" for i in range(6)]),
        "observed_at": np.arange(len(x)),
        "signature": np.arange(len(x), dtype=np.uint64),
    }
    # Synthetic schema has only one corruption group unless explicitly split.
    from unittest.mock import patch

    with patch(
        "fedxmag_next.record_prequential.corruption_groups", return_value=[[0, 1, 2], [3, 4, 5]]
    ):
        base = build_source(
            data,
            np.arange(128),
            np.arange(128, 256),
            Namespace(max_reference=128, seeds=[17], epochs=1),
        )
    before = {k: v.clone() for k, v in base["seed_models"][0]["model"].items()}
    rkeys = {"source": ids[:128], "destination": ids[:128][::-1].copy()}
    skeys = {"source": ids[128:], "destination": ids[128:][::-1].copy()}
    state = fit_entity_state(
        base, torch.from_numpy(x[:128]), torch.from_numpy(x[128:]), rkeys, skeys
    )
    for k, value in before.items():
        assert torch.equal(value, base["seed_models"][0]["model"][k])
    for bank in state["banks"].values():
        assert set(bank) == {"-1", "0", "1"}
    path = tmp_path / "entity.pt"
    torch.save(state, path)
    query = torch.from_numpy(x[:9])
    qkeys = {scope: values[:9] for scope, values in rkeys.items()}
    fitted = EntitySupportDetector(state)
    scores = fitted.score(query, qkeys)
    replay = EntitySupportDetector.load(path).score(query, qkeys)
    assert len(scores) == 12
    for name, score in scores.items():
        assert torch.isfinite(score).all()
        torch.testing.assert_close(score, replay[name], rtol=0, atol=0)
    unknown = {scope: np.full(9, 123, dtype=np.uint64) for scope in qkeys}
    fallback = fitted.score(query, unknown)
    for prefix in ("raw", "deep"):
        for modes in (1, 8):
            suffix = "_ensemble" if prefix == "deep" else ""
            global_score = fallback[f"{prefix}_entity_global_m{modes}{suffix}"]
            for scope in ("source", "destination"):
                torch.testing.assert_close(
                    fallback[f"{prefix}_entity_{scope}_m{modes}{suffix}"], global_score
                )
    empty = fitted.score(query[:0], {k: v[:0] for k, v in qkeys.items()})
    assert all(v.shape == (0,) for v in empty.values())
    with pytest.raises(ValueError):
        fitted.score(query, {k: v[:2] for k, v in qkeys.items()})
