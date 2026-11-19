from argparse import Namespace

import numpy as np
import torch
from test_context_association import observed_protocols

from fedxmag_next.cohort_energy import CohortEnergyBank, CohortEnergyDetector, empirical_typicality
from fedxmag_next.context_association import cohort_ids
from fedxmag_next.daily_detector import DailyDeepEnergyDetector
from fedxmag_next.geometry import ConditionalMahalanobis
from fedxmag_next.record_data import FEATURES
from fedxmag_next.record_prequential import build_source


def test_cohort_geometry_routes_observed_cohort_and_falls_back_without_refitting():
    torch.set_num_threads(2)
    torch.manual_seed(326)
    reference_ids = torch.tensor([0] * 128 + [1] * 128 + [2] * 12)
    selection_ids = torch.tensor([0] * 64 + [1] * 64 + [2] * 8)
    reference = torch.randn(len(reference_ids), 4) + 7 * reference_ids[:, None]
    selection = torch.randn(len(selection_ids), 4) + 7 * selection_ids[:, None]
    bank = CohortEnergyBank.fit(reference, selection, reference_ids, selection_ids, modes=1)
    assert set(bank.entries) == {"-1", "0", "1"}
    query = torch.randn(4, 4)
    ids = torch.tensor([0, 1, 2, 99])
    actual = bank.score(query, ids, conditioned=True)
    for i, group in enumerate(ids):
        entry = bank.entries.get(str(int(group)), bank.entries["-1"])
        support = ConditionalMahalanobis.from_state_dict(entry["geometry"])
        low, center, high = entry["quantiles"]
        energy = support.score(query[i : i + 1])[0][0]
        expected = (energy - center) / (high - center).clamp_min(1e-6)
        torch.testing.assert_close(actual["upper"][i], expected)
        expected = torch.maximum(expected, (center - energy) / (center - low).clamp_min(1e-6))
        torch.testing.assert_close(actual["two_sided"][i], expected)
    global_scores = bank.score(query, ids, conditioned=False)
    assert actual["upper"][2] == global_scores["upper"][2]
    assert actual["upper"][3] == global_scores["upper"][3]
    torch.testing.assert_close(
        bank.score(query.flip(0), ids.flip(0), conditioned=True)["two_sided"].flip(0),
        actual["two_sided"],
    )


def test_two_sided_energy_detects_lower_tail_and_uses_only_frozen_quantiles(tmp_path):
    geometry = ConditionalMahalanobis.fit(
        torch.tensor([[-1.0], [1.0]]).repeat(32, 1), max_components=1
    )
    bank = CohortEnergyBank(
        {
            "-1": {
                "geometry": geometry.state_dict(),
                "quantiles": torch.tensor([1.0, 2.0, 4.0]).double(),
            }
        }
    )
    query = torch.tensor([[0.0], [10.0]])
    score = bank.score(query, torch.zeros(2, dtype=torch.long), conditioned=True)
    assert score["upper"][0] < 0
    assert score["two_sided"][0] == 2
    assert score["two_sided"][1] == score["upper"][1]
    path = tmp_path / "bank.pt"
    torch.save(bank.entries, path)
    restored = CohortEnergyBank(torch.load(path, weights_only=True))
    replay = restored.score(query, torch.zeros(2, dtype=torch.long), conditioned=True)
    assert all(torch.equal(score[key], replay[key]) for key in score)


def test_cohort_detector_restores_all_raw_and_neural_scores(tmp_path):
    torch.set_num_threads(2)
    values = np.random.default_rng(91).normal(size=(192, len(FEATURES))).astype(np.float32)
    observed_protocols(values)
    data = {
        "features": values,
        "feature_names": np.array(FEATURES),
        "observed_at": np.arange(192),
        "signature": np.arange(192, dtype=np.uint64),
    }
    base = build_source(
        data,
        np.arange(128),
        np.arange(128, 192),
        Namespace(max_reference=128, seeds=[17], epochs=1),
    )
    detector = DailyDeepEnergyDetector(base)
    query = torch.from_numpy(values)
    normalized = detector.normalize_features(query)
    with torch.no_grad():
        embedded = detector.models[0].layers[:-1](normalized)
    item = base["seed_models"][0]
    embedded = (embedded - item["embedding_mean"]) / item["embedding_scale"]
    ids = cohort_ids(query)
    banks = {}
    for name, x in (("raw", normalized), ("seed_17", embedded)):
        for modes in (1, 8):
            banks[f"{name}_m{modes}"] = CohortEnergyBank.fit(
                x[:128], x[128:], ids[:128], ids[128:], modes
            ).entries
    state = {"format_version": 1, "base_encoder": base, "banks": banks}
    path = tmp_path / "cohort.pt"
    torch.save(state, path)
    scores = CohortEnergyDetector(state).score(query[:9])
    replay = CohortEnergyDetector.load(path).score(query[:9])
    assert len(scores) == 16
    for name in scores:
        assert torch.isfinite(scores[name]).all()
        torch.testing.assert_close(scores[name], replay[name], rtol=0, atol=0)
    empty = CohortEnergyDetector.load(path).score(query[:0])
    assert set(empty) == set(scores)
    assert all(value.shape == (0,) for value in empty.values())
    rank_state = {**state, "score_style": "rank"}
    rank_path = tmp_path / "rank_cohort.pt"
    torch.save(rank_state, rank_path)
    ranked = CohortEnergyDetector(rank_state).score(query[:9])
    rank_replay = CohortEnergyDetector.load(rank_path).score(query[:9])
    assert len(ranked) == 16 and all("rank" in name for name in ranked)
    for name in ranked:
        assert torch.isfinite(ranked[name]).all()
        torch.testing.assert_close(ranked[name], rank_replay[name], rtol=0, atol=0)


def test_empirical_two_sided_ranks_balance_both_extremes_and_respect_ties():
    reference = torch.tensor([1.0, 2.0, 2.0, 4.0]).double()
    query = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).double()
    scores = empirical_typicality(query, reference)
    upper = torch.tensor([1.0, 1.0, 0.8, 0.4, 0.4, 0.2]).double()
    two_sided = torch.tensor([0.4, 0.8, 1.0, 0.8, 0.8, 0.4]).double()
    torch.testing.assert_close(scores["upper"], -upper.log())
    torch.testing.assert_close(scores["two_sided"], -two_sided.log())
    assert scores["two_sided"][0] == scores["two_sided"][-1]
    transformed = empirical_typicality(query.exp(), reference.exp())
    assert all(torch.equal(scores[key], transformed[key]) for key in scores)
    perturbed = query.clone()
    perturbed[1:3] += 1e-8
    rounded = empirical_typicality(perturbed, reference)
    assert all(torch.equal(scores[key], rounded[key]) for key in scores)
