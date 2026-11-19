"""Conditional normal-support graph anomaly detection.

The public API intentionally keeps model fitting, threshold calibration, and
trusted-normal commissioning as separate operations.
"""

from .contracts import GraphWindow
from .detector import ConditionalSupportDetector
from .geometry import ConditionalMahalanobis
from .model import CausalEdgeGraphEncoder, EncoderConfig

__all__ = [
    "CausalEdgeGraphEncoder",
    "ConditionalMahalanobis",
    "ConditionalSupportDetector",
    "EncoderConfig",
    "GraphWindow",
]
