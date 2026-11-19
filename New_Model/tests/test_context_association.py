from argparse import Namespace

import numpy as np
import torch

from fedxmag_next.context_association import ContextPairSampler, cohort_ids
from fedxmag_next.contrastive_energy import DependencyEnergy
from fedxmag_next.daily_detector import DailyDeepEnergyDetector
from fedxmag_next.record_data import FEATURES, PROTOCOLS, SERVICES
from fedxmag_next.record_prequential import build_source
from fedxmag_next.temporal_context import CONTEXT_FEATURES


def test_matched_negatives_always_change_history_even_for_dominant_duplicate_class():
    values = torch.tensor(
        [
            [10.0, 20.0, 0.0, 0.0],
            [11.0, 21.0, 0.0, 0.0],
            [12.0, 22.0, 0.0, 0.0],
            [13.0, 23.0, 1.0, 2.0],
            [14.0, 24.0, 3.0, 4.0],
            [15.0, 25.0, 3.0, 4.0],
            [16.0, 26.0, 5.0, 6.0],
            [17.0, 27.0, 7.0, 8.0],
        ]
    )
    cohorts = torch.tensor([0, 0, 0, 0, 1, 1, 2, 2])
    sampler = ContextPairSampler(values, cohorts, [2, 3])
    assert sampler.eligible == 6
    for seed in range(10):
        ids, donors = sampler.sample(torch.Generator().manual_seed(seed))
        assert torch.equal(cohorts[ids], cohorts[donors])
        assert torch.any(values[ids, 2:] != values[donors, 2:], dim=1).all()
        negative = sampler.negative(ids, donors)
        assert torch.equal(negative[:, :2], values[ids, :2])
        assert not torch.isin(ids, torch.tensor([4, 5])).any()


def observed_protocols(features):
    columns = [FEATURES.index(f"protocol_{p}") for p in (*PROTOCOLS, "other")]
    features[:, columns] = 0
    features[:, columns[0]] = 1
    columns = [FEATURES.index(f"service_{p}") for p in (*SERVICES, "other")]
    features[:, columns] = 0
    features[np.arange(len(features)), np.array(columns)[np.arange(len(features)) % 2]] = 1


def test_cohorts_use_observed_protocol_and_service_categories():
    values = np.zeros((6, len(FEATURES)), dtype=np.float32)
    observed_protocols(values)
    assert cohort_ids(torch.from_numpy(values)).tolist() == [0, 1, 0, 1, 0, 1]


def test_association_source_restores_native_density_ratio_and_normal_geometry(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(81)
    names = FEATURES + CONTEXT_FEATURES
    features = rng.normal(size=(256, len(names))).astype(np.float32)
    observed_protocols(features)
    data = {
        "features": features,
        "feature_names": np.array(names),
        "observed_at": np.arange(256),
        "signature": np.arange(256, dtype=np.uint64),
    }
    state = build_source(
        data,
        np.arange(192),
        np.arange(192, 256),
        Namespace(max_reference=192, seeds=[17], epochs=1, objective="association"),
    )
    assert state["seed_models"][0]["history"][0]["train_changed_context_pairs"] == 192
    path = tmp_path / "association.pt"
    torch.save(state, path)
    restored = DailyDeepEnergyDetector.load(path)
    model = DependencyEnergy(len(names))
    item = state["seed_models"][0]
    model.load_state_dict(item["model"])
    query = torch.from_numpy(features[:9])
    expected = model(restored.normalize_features(query)).double() - item["native_score_center"]
    expected = expected / item["native_score_scale"] + 1
    actual = restored.score_all(query)
    for modes in (1, 8):
        torch.testing.assert_close(actual[modes]["association"], expected)
        assert all(torch.isfinite(v).all() for v in actual[modes].values())
