from argparse import Namespace

import numpy as np
import torch

from fedxmag_next.daily_detector import DailyDeepEnergyDetector
from fedxmag_next.record_data import FEATURES
from fedxmag_next.record_prequential import build_source


def test_normal_warm_adaptation_preserves_input_units_and_has_identical_raw_controls():
    torch.set_num_threads(2)
    features = np.random.default_rng(157).normal(size=(320, len(FEATURES))).astype(np.float32)
    features[:192, 0] = 0
    features[192:, 0] = 1
    data = {
        "features": features,
        "feature_names": np.array(FEATURES),
        "signature": np.arange(320, dtype=np.uint64),
        "observed_at": np.arange(320),
    }
    args = Namespace(max_reference=192, seeds=[17], epochs=1)
    initial = build_source(data, np.arange(128), np.arange(128, 192), args)
    reference = np.r_[np.arange(96), np.arange(192, 288)]
    selected = np.r_[np.arange(128, 160), np.arange(288, 320)]
    frozen = build_source(
        data,
        reference,
        selected,
        Namespace(max_reference=192, seeds=[17], epochs=0),
        initial_state=initial,
    )
    tuned = build_source(data, reference, selected, args, initial_state=initial)
    assert torch.equal(initial["input_mean"], frozen["input_mean"])
    assert torch.equal(initial["input_scale"], tuned["input_scale"])
    assert tuned["seed_models"][0]["history"][0]["epoch"] == 0
    for name, value in initial["seed_models"][0]["model"].items():
        assert torch.equal(value, frozen["seed_models"][0]["model"][name])
    query = torch.from_numpy(features[300:309])
    a, b = (
        DailyDeepEnergyDetector(frozen).score_all(query),
        DailyDeepEnergyDetector(tuned).score_all(query),
    )
    for modes in (1, 8):
        for key in ("raw_knn", "raw_mahalanobis"):
            torch.testing.assert_close(a[modes][key], b[modes][key], rtol=0, atol=0)
        assert all(torch.isfinite(value).all() for value in b[modes].values())
