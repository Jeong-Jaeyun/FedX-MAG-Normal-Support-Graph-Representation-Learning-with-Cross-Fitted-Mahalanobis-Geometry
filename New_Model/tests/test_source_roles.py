import numpy as np

from fedxmag_next.source_only import source_roles


def test_both_source_orders_keep_roles_disjoint_and_purge_shared_context():
    data = {"timestamp": np.arange(1000) * 30, "observed_at": np.arange(1000) * 30 + 1}
    normal = np.arange(1000)
    for order in ("train_selection_calibration", "calibration_selection_train"):
        roles = source_roles(data, normal, order)
        sequence = order.split("_")
        for previous, following in zip(sequence, sequence[1:], strict=False):
            assert data["timestamp"][roles[following]].min() > (
                data["observed_at"][roles[previous]].max() + 300
            )
            assert not np.intersect1d(roles[previous], roles[following]).size
