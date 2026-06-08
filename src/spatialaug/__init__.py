"""spatialaug — geospatial imputation & augmentation framework.

Quickstart:
    >>> from spatialaug import (
    ...     # Imputers (9)
    ...     KrigingImputer, UniversalKrigingImputer, GBMImputer,
    ...     KrigingPriorRegression, TabPFNImputer,
    ...     MeanImputer, IDWImputer, KNNImputer,
    ...     # Augmenter — synthetic point generation
    ...     Augmenter,
    ...     # Privacy emulation (ОФД-style MNAR)
    ...     ValueBasedMask,
    ...     # Cross-region evaluation
    ...     TransferEvaluator,
    ...     # Spatial CV + metrics
    ...     SpatialBlockCV, compute_error_suite, morans_i,
    ... )
"""

# Imputers
from spatialaug.imputers import (
    GBMImputer,
    IDWImputer,
    Imputer,
    KNNImputer,
    KrigingImputer,
    KrigingPriorRegression,
    MeanImputer,
    TabPFNImputer,
    UniversalKrigingImputer,
)

# Augmenters
from spatialaug.augmenters import Augmenter

# Privacy emulation
from spatialaug.privacy import ValueBasedMask

# Cross-region evaluation
from spatialaug.transfer import TransferEvaluator, TransferResult

# Metrics
from spatialaug.metrics import (
    SpatialBlockCV,
    by_target_quartile,
    compute_error_suite,
    morans_i,
    morans_preservation_ratio,
)

# Benchmark utilities
from spatialaug.benchmark import MissingnessSimulator

# Profiling
from spatialaug.utils import MethodProfiler, profile_fit_transform

__version__ = "0.0.4"
__all__ = [
    "Imputer",
    "MeanImputer",
    "IDWImputer",
    "KNNImputer",
    "KrigingImputer",
    "UniversalKrigingImputer",
    "GBMImputer",
    "KrigingPriorRegression",
    "TabPFNImputer",
    "Augmenter",
    "ValueBasedMask",
    "TransferEvaluator",
    "TransferResult",
    "SpatialBlockCV",
    "compute_error_suite",
    "by_target_quartile",
    "morans_i",
    "morans_preservation_ratio",
    "MissingnessSimulator",
    "MethodProfiler",
    "profile_fit_transform",
]
