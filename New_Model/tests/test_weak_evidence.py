import numpy as np
import pytest

from fedxmag_next.weak_evidence import CausalEvidenceMemory


def test_additive_memory_uses_original_evidence_once_per_key_per_window():
    peak, summed = (CausalEvidenceMemory(1, mode) for mode in ("peak", "sum"))
    for model in (peak, summed):
        np.testing.assert_array_equal(model.predict(60, [7, 7], [3, 3], [20, 30]), [3, 3])
        model.commit([0, 1])
        assert len(model.states[7]) == 1
        carried = model.predict(120, [7], [2], [100])
        assert carried[0] > 2
        model.commit([0])  # must store original2, not the carried output
        assert model.states[7][-1] == (100, 1)
    assert summed.predict(180, [7], [1], [160])[0] > peak.predict(180, [7], [1], [160])[0]


def test_evidence_checkpoint_expiry_and_key_budget_are_exact():
    model = CausalEvidenceMemory(1, "sum", horizon=120, maximum_keys=1)
    model.predict(60, [1, 2], [3, 4], [40, 50])
    with pytest.raises(ValueError, match="committed"):
        model.state_dict()
    model.commit([0, 1])
    assert list(model.states) == [2]
    restored = CausalEvidenceMemory.from_state_dict(model.state_dict())
    np.testing.assert_array_equal(
        model.predict(240, [2], [1], [230]), restored.predict(240, [2], [1], [230])
    )
    assert not model.states
