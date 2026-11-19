"""M1: joint reconstruction and cross-fitted Mahalanobis normal-support learning.

Kept outside ``models/`` and ``training/`` so the frozen M0 fingerprints are
unchanged by its addition.
"""

from fedxmag.m1.covariance import (
    ESTIMATORS,
    CovarianceEstimate,
    estimate_covariance,
    ledoit_wolf_shrinkage,
    linear_cka,
    mahalanobis_energy,
    oas_shrinkage,
    spectral_diagnostics,
)
from fedxmag.m1.objective import NormalSupportSettings, normal_support_loss
from fedxmag.m1.training import clean_view, embed_clean, train_m1

__all__ = [
    "ESTIMATORS",
    "CovarianceEstimate",
    "NormalSupportSettings",
    "clean_view",
    "embed_clean",
    "estimate_covariance",
    "ledoit_wolf_shrinkage",
    "linear_cka",
    "mahalanobis_energy",
    "normal_support_loss",
    "oas_shrinkage",
    "spectral_diagnostics",
    "train_m1",
]
