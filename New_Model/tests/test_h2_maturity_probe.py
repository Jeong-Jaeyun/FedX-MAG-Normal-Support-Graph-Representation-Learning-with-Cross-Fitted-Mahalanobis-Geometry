import importlib.util
from pathlib import Path

import pytest
import torch

from fedxmag_next.h1_classical_features import SourcePCA

spec = importlib.util.spec_from_file_location(
    "probe_h2_maturity", Path(__file__).resolve().parents[1] / "scripts/probe_h2_maturity.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_source_continuation_records_gradients_and_restorable_checkpoint():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        generator = torch.Generator().manual_seed(71)
        raw = torch.randn(96, 8, generator=generator, dtype=torch.float64)
        raw[:, 1] = raw[:, 0] + 0.1 * raw[:, 1]
        valid = torch.randn(40, 8, generator=generator, dtype=torch.float64)
        valid[:, 1] = valid[:, 0] + 0.1 * valid[:, 1]
        adapter = SourcePCA.fit(raw)
        model = probe.construct("mlp32")
        result = probe.continue_source(
            model, adapter, raw, valid, [[i] for i in range(8)], epoch_offset=30, epochs=2
        )
        assert [h["absolute_epoch"] for h in result["history"]] == [30, 31, 32]
        assert result["best_bce"] <= result["initial_bce"]
        assert set(result["history"][1]["unclipped_gradient_rms"]) == {"encoder", "readout"}
        assert result["history"][1]["unclipped_gradient_rms"]["encoder"] > 0
        assert result["source_feature_summary_selected"]["finite"]
        restored = probe.construct("mlp32")
        restored.load_state_dict(result["selected_state"])
        torch.testing.assert_close(
            model(adapter.transform(valid)), restored(adapter.transform(valid))
        )
    finally:
        torch.set_num_threads(previous)


def test_probe_rejects_empty_continuation_and_unknown_models():
    with pytest.raises(ValueError, match="Unsupported"):
        probe.construct("unknown")
    with pytest.raises(ValueError, match="Positive"):
        probe.continue_source(None, None, None, None, None, epochs=0)


def test_loader_never_requests_target_label_or_census_arrays():
    import inspect

    source = inspect.getsource(probe.load_source)
    assert 'payload["label"]' not in source
    assert 'payload["observed_at"]' not in source
    assert 'payload["features"]' in source
    assert "H2Data(" not in source
