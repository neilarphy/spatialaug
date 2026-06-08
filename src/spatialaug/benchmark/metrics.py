"""Robustness metrics for the benchmark.

These score not only how well an imputer reconstructs missing values
on a single run, but how stable it is across heterogeneous
conditions 
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def stability_score(metric_values: Sequence[float]) -> float:
    """Stability score = 1 - std(metric) / |mean(metric)|.

    Higher (closer to 1) means the metric varies less across the
    supplied conditions (e.g. across missingness mechanisms or
    seeds). May be negative when std exceeds |mean|.

    Parameters
    ----------
    metric_values : sequence of float
        Values of the metric across conditions (e.g. MAE under
        mcar / diffuse_mnar / focused_mnar).

    Returns
    -------
    float
        Stability score. NaN if the input is empty or mean is zero.
    """
    arr = np.asarray(list(metric_values), dtype=float)
    
    if arr.size == 0:
        return float("nan")
    
    mean = float(arr.mean())
    
    if mean == 0:
        return float("nan")
    
    return 1.0 - float(arr.std(ddof=0)) / abs(mean)


def degradation_slope(
    densities: Sequence[float], metrics: Sequence[float]
) -> float:
    """Slope of the linear fit ``metric ~ a + slope * density``.

    The more negative the slope, the more the method degrades as
    data density drops. For MAE / RMSE, a negative slope versus
    increasing density simply means quality improves with more
    data, which is expected.

    Returns
    -------
    float
        The fitted slope.
    """
    
    if len(densities) != len(metrics):
        raise ValueError(
            f"densities and metrics must have same length, "
            f"got {len(densities)} and {len(metrics)}"
        )
    
    if len(densities) < 2:
        raise ValueError("Need at least 2 points for slope")
    
    x = np.asarray(densities, dtype=float)
    y = np.asarray(metrics, dtype=float)
    slope, _ = np.polyfit(x, y, deg=1)
    
    return float(slope)


def delta_f1(f1_augmented: float, f1_original: float) -> float:
    """Delta F1 = F1(augmented) - F1(original).

    Positive: augmentation helps downstream classification.
    Negative: it hurts.
    """
    
    return float(f1_augmented) - float(f1_original)
