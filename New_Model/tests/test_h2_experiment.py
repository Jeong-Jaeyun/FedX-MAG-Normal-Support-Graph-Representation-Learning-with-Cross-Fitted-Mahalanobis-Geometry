"""Checkpoint selection and self-contained H2 scalar/support/fusion artifacts."""

import pytest
import torch

from fedxmag_next.geometry import ConditionalMahalanobis
from fedxmag_next.h1_classical_features import SourcePCA
from fedxmag_next.h1_experiment import H1Detector
from fedxmag_next.h2_data import negative_rows
from fedxmag_next.h2_experiment import H2Detector, make_model, train_model, validation_bce
from fedxmag_next.record_data import FEATURES, NUMERIC
from fedxmag_next.record_experiment import corruption_groups


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def normal_pool(n, seed):
    names = list(FEATURES)
    generator = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, len(names), dtype=torch.float64)
    for name in NUMERIC:
        x[:, names.index("log_" + name)] = torch.rand(n, generator=generator) * 3
        x[:, names.index("available_" + name)] = 1
    for name in ("protocol_tcp", "service_http", "state_SF", "destination_port_80"):
        x[:, names.index(name)] = 1
    return x, names


def test_restored_training_checkpoint_matches_minimum_normal_selection_loss():
    train, names = normal_pool(96, 21)
    selection, _ = normal_pool(48, 22)
    adapter = SourcePCA.fit(train)
    groups = corruption_groups(names)
    model = make_model("mlp32", len(names), 17)
    result = train_model(
        model,
        "mlp32",
        adapter,
        train,
        selection,
        groups,
        names,
        "conditional",
        17,
        epochs=4,
    )
    losses = [entry["selection_synthetic_bce"] for entry in result["history"]]
    assert result["selected_epoch"] == losses.index(min(losses))
    actual = validation_bce(
        model,
        adapter.transform(selection),
        adapter.transform(negative_rows(selection, groups, names, "conditional", 7189)),
    )
    assert actual == pytest.approx(min(losses), abs=1e-14)
    assert result["train_corruption_audit"]["all_groups_are_normal_donor_groups"]
    assert result["parameter_count"] == 1377


def test_training_rejects_no_epoch_before_checkpoint_audit():
    train, names = normal_pool(32, 21)
    with pytest.raises(ValueError, match="at least one epoch"):
        train_model(
            make_model("mlp32", len(names), 17),
            "mlp32",
            SourcePCA.fit(train),
            train,
            train,
            corruption_groups(names),
            names,
            "original",
            17,
            epochs=0,
        )


def test_frozen_h2_fused_artifact_restores_query_scores_and_batching(tmp_path):
    train, names = normal_pool(96, 21)
    query, _ = normal_pool(35, 23)
    adapter = SourcePCA.fit(train)
    unit = {"center": 0.3, "scale": 2.1}
    raw_geometry = ConditionalMahalanobis.fit(adapter.normalize(train), max_components=1)
    raw_state = {
        "format_version": 1,
        "kind": "raw",
        "adapter": adapter.state_dict(),
        "seeds": [
            {
                "map": {},
                "mean": torch.zeros(len(names)),
                "scale": torch.ones(len(names)),
                "supports": {
                    str(m): {"geometry": raw_geometry.state_dict(), **unit} for m in (1, 8)
                },
            }
        ],
        "ensemble_units": {str(m): unit for m in (1, 8)},
    }
    model = make_model("mlp32", len(names), 17)
    with torch.no_grad():
        fit = model.embed(adapter.transform(train))
    mean, scale = fit.mean(0), fit.std(0, unbiased=False).clamp_min(1e-6)
    geometry = ConditionalMahalanobis.fit((fit - mean) / scale, max_components=1)
    state = {
        "format_version": 1,
        "kind": "mlp32",
        "feature_names": names,
        "adapter": adapter.state_dict(),
        "raw_bypass_state": raw_state,
        "seeds": [
            {
                "seed": 17,
                "model": model.state_dict(),
                "feature_mean": mean,
                "feature_scale": scale,
                "geometry": geometry.state_dict(),
                "units": {"scalar": unit, "support": unit},
            }
        ],
        "ensemble_units": {"scalar": unit, "support": unit},
    }
    path = tmp_path / "model.pt"
    torch.save(state, path)
    detector = H2Detector(torch.load(path, weights_only=True))
    actual = detector.score_with_raw(query)
    with torch.no_grad():
        hidden = model.embed(adapter.transform(query))
        expected_heads = {
            "scalar": model.readout(hidden).squeeze(-1),
            "support": geometry.score((hidden - mean) / scale)[0],
        }
    raw = H1Detector(raw_state).score(query)[8]
    for name, value in expected_heads.items():
        once = (value - unit["center"]) / unit["scale"] + 1
        expected = (once - unit["center"]) / unit["scale"] + 1
        torch.testing.assert_close(actual[name], expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(actual["raw_plus_" + name], torch.maximum(expected, raw))
    chunked = [detector.score_with_raw(part) for part in query.split(9)]
    for name in actual:
        torch.testing.assert_close(torch.cat([part[name] for part in chunked]), actual[name])
