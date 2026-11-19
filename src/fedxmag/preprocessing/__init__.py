"""Dataset adapters and leakage-safe static graph construction."""

from fedxmag.preprocessing.events import (
    CanonicalEvent,
    EventReadResult,
    IngestionReport,
    XIIoTIDColumns,
    read_xiiotid_events,
)
from fedxmag.preprocessing.graph_builder import (
    build_static_graphs,
    chronological_window_aligned_split,
)
from fedxmag.preprocessing.scaling import FeatureStandardizer

__all__ = [
    "CanonicalEvent",
    "FeatureStandardizer",
    "EventReadResult",
    "IngestionReport",
    "XIIoTIDColumns",
    "build_static_graphs",
    "chronological_window_aligned_split",
    "read_xiiotid_events",
]
