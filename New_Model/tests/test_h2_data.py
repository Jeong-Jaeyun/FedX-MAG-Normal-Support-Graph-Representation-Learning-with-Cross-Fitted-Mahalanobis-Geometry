import inspect

import pytest
import torch

from fedxmag_next.contrastive_energy import corrupt
from fedxmag_next.h1_classical_features import SourcePCA
from fedxmag_next.h2_data import conditioning_columns, corruption_audit, negative_rows
from fedxmag_next.record_data import FEATURES, NUMERIC
from fedxmag_next.record_experiment import corruption_groups


def normal_pool(n=32):
    names = list(FEATURES)
    x = torch.zeros(n, len(names), dtype=torch.float64)
    for name in NUMERIC:
        x[:, names.index("log_" + name)] = torch.arange(n).double() + 1
        x[:, names.index("available_" + name)] = 1
    for name in ("protocol_tcp", "service_http", "state_SF", "destination_port_80"):
        x[:, names.index(name)] = 1
    return x, names, corruption_groups(names)


def test_original_is_historical_corruption_on_the_same_pool():
    x, names, groups = normal_pool()
    expected = corrupt(x, groups, torch.Generator().manual_seed(42))
    actual = negative_rows(x, groups, names, "original", 42)
    assert torch.equal(actual, expected)
    audit = corruption_audit(x, actual, groups, names)
    assert audit["changed_row_fraction"] > 0
    assert all(audit["numeric_availability_valid"].values())


def test_conditional_exact_cohort_preserves_metadata_and_donor_groups():
    x, names, groups = normal_pool()
    x[16:, names.index("service_http")] = 0
    x[16:, names.index("service_dns")] = 1
    x[::3, names.index("available_duration")] = 0
    x[::3, names.index("log_duration")] = 0
    before = x.clone()
    actual = negative_rows(x, groups, names, "conditional", 71)
    assert torch.equal(x, before)
    assert torch.equal(actual, negative_rows(x, groups, names, "conditional", 71))
    columns, _ = conditioning_columns(names)
    assert torch.equal(x[:, columns], actual[:, columns])
    audit = corruption_audit(x, actual, groups, names)
    assert audit["changed_row_fraction"] > 0
    assert audit["all_groups_are_normal_donor_groups"]


def test_singleton_fallback_never_relaxes_protocol_service_or_group_availability():
    x, names, groups = normal_pool(4)
    # Four exact singleton cohorts. Row0 can borrow duration from row1 by
    # relaxing unrelated bytes availability, but never DNS row2 or UDP row3.
    x[0, names.index("available_src_bytes")] = 0
    x[0, names.index("log_src_bytes")] = 0
    x[2, names.index("service_http")] = 0
    x[2, names.index("service_dns")] = 1
    x[3, names.index("protocol_tcp")] = 0
    x[3, names.index("protocol_udp")] = 1
    columns, _ = conditioning_columns(names)
    sampled = set()
    for seed in range(24):
        actual = negative_rows(x, groups, names, "conditional", seed)
        assert torch.equal(x[:, columns], actual[:, columns])
        sampled.add(float(actual[0, names.index("log_duration")]))
        assert torch.equal(actual[2], x[2])
        assert torch.equal(actual[3], x[3])
        corruption_audit(x, actual, groups, names)
    assert sampled == {1.0, 2.0}


def test_corruption_precedes_frozen_pca_without_refitting_or_labels():
    x, names, groups = normal_pool(64)
    x[:, names.index("log_dst_port")] = torch.arange(len(x)).double() ** 2
    adapter = SourcePCA.fit(x)
    before = {k: v.clone() for k, v in adapter.state_dict().items()}
    actual = negative_rows(x, groups, names, "conditional", 9)
    assert adapter.transform(actual).shape == (len(x), 8)
    assert set(inspect.signature(negative_rows).parameters) == {
        "values",
        "groups",
        "names",
        "policy",
        "seed",
    }
    for key, value in before.items():
        assert torch.equal(adapter.state_dict()[key], value)


@pytest.mark.parametrize("policy", ["original", "conditional"])
def test_corruption_rejects_partial_or_latent_schema(policy):
    x, names, groups = normal_pool()
    with pytest.raises(ValueError, match="partition"):
        negative_rows(x, groups[:-1], names, policy, 17)
    with pytest.raises(ValueError, match="original feature"):
        negative_rows(x[:, :8], groups, names, policy, 17)


def test_audit_detects_broken_numeric_availability():
    x, names, groups = normal_pool()
    negative = x.clone()
    negative[0, names.index("available_duration")] = 0
    with pytest.raises(ValueError, match="numeric"):
        corruption_audit(x, negative, groups, names)
