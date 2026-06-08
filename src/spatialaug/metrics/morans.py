"""Moran's I spatial autocorrelation metric.

Standard formula:
    I = (N / W) * sum_i sum_j w_ij (x_i - mean)(x_j - mean) / sum_i (x_i - mean)^2

For row-normalized KNN weights (W = N) this reduces to:
    I = sum_i (x_i - mean) * mean_neighbors(x_i - mean) / sum_i (x_i - mean)^2

Interpretation:
- I close to +1: strong positive autocorrelation (neighbors are similar).
- I close to  0: spatial randomness.
- I close to -1: negative autocorrelation (neighbors differ, checkerboard).

Used to evaluate imputation: ratio = Moran_I(imputed) / Moran_I(observed).
- ratio close to 1: the method preserves spatial structure.
- ratio < 1: the method over-smooths (typical for kriging with a
  large nugget).
- ratio > 1: the method amplifies structure (suspicious — check for
  artefacts).
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def morans_i(
    coords: np.ndarray,
    values: np.ndarray,
    k_neighbors: int = 8,
    n_permutations: int = 0,
    random_state: int = 42,
) -> tuple[float, float]:
    """Compute Moran's I and an optional permutation-based p-value.

    Parameters
    ----------
    coords : (N, 2) array
        Coordinates, e.g. lat/lon or x/y.
    values : (N,) array
        Target values aligned with coords.
    k_neighbors : int, default=8
        Number of nearest neighbors used to build the row-normalized
        weight matrix (self is excluded).
    n_permutations : int, default=0
        Number of permutations for the p-value. 0 skips the test.
    random_state : int, default=42
        Seed for the permutation shuffle.

    Returns
    -------
    (i, p_value) : tuple of float
        i is Moran's I (NaN if variance is zero or N < k+1).
        p_value is the two-sided permutation p-value (NaN when
        n_permutations is 0 or variance is zero).
    """
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < k_neighbors + 1:
        return float("nan"), float("nan")

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    neighbor_idx = idx[:, 1:]  

    x = values - values.mean()
    var = (x * x).sum()
    if var < 1e-12:
        return 0.0, float("nan")

    neighbor_mean = x[neighbor_idx].mean(axis=1)
    i = float((x * neighbor_mean).sum() / var)

    if n_permutations <= 0:
        return i, float("nan")

    rng = np.random.default_rng(random_state)
    perm_stats = np.empty(n_permutations)
    for k in range(n_permutations):
        perm = rng.permutation(values) - values.mean()
        perm_var = (perm * perm).sum()
        if perm_var < 1e-12:
            perm_stats[k] = 0.0
        else:
            perm_stats[k] = (perm * perm[neighbor_idx].mean(axis=1)).sum() / perm_var
    p_value = float(np.mean(np.abs(perm_stats) >= np.abs(i)))
    return i, p_value


def morans_preservation_ratio(
    coords: np.ndarray,
    original_values: np.ndarray,
    imputed_values: np.ndarray,
    k_neighbors: int = 8,
) -> dict[str, float]:
    """Spatial-autocorrelation preservation metric for imputation.

    Computes Moran's I on both the original and the imputed value
    arrays (both aligned with the same coords) and reports the
    ratio and delta. The caller is responsible for choosing what
    "original" means: typically the full ground-truth vector when
    benchmarking under synthetic masking.

    Returns
    -------
    dict
        morans_original — I computed on original_values.
        morans_imputed  — I computed on imputed_values.
        ratio           — imputed / original (NaN if original I is
                          near zero).
        delta           — imputed - original.
    """
    
    i_orig, _ = morans_i(coords, original_values, k_neighbors=k_neighbors)
    i_imp, _ = morans_i(coords, imputed_values, k_neighbors=k_neighbors)
    ratio = i_imp / i_orig if abs(i_orig) > 1e-12 else float("nan")
    
    return {
        "morans_original": i_orig,
        "morans_imputed": i_imp,
        "ratio": ratio,
        "delta": i_imp - i_orig,
    }
