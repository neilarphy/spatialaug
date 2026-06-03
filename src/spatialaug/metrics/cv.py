"""Spatial block cross-validation."""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd


def _km_to_deg(center_lat: float, km: float) -> tuple[float, float]:
    """Convert km to degrees of (lat, lon) at given latitude.

    1 degree of latitude ≈ 111 km globally (constant).
    1 degree of longitude ≈ 111 km × cos(lat).
    """
    deg_lat = km / 111.0
    deg_lon = km / (111.0 * max(math.cos(math.radians(center_lat)), 1e-6))
    return deg_lat, deg_lon


class SpatialBlockCV:
    """K-fold cross-validator с пространственным разбиением.

    Parameters
    ----------
    n_splits : int, default=5
        Количество фолдов.
    block_size_km : float, optional
        Размер блока в км. Если None — вычисляется из bbox данных:
        `bbox_diagonal / (n_splits × 2)`.
    random_state : int, default=42
        Seed для перемешивания блоков между фолдами.

    Examples
    --------
    >>> cv = SpatialBlockCV(n_splits=5, block_size_km=10)
    >>> for train_idx, test_idx in cv.split(df, lat_col="lat", lon_col="lon"):
    ...     X_train = df.iloc[train_idx]
    ...     X_test = df.iloc[test_idx]
    """

    def __init__(
        self,
        n_splits: int = 5,
        block_size_km: float | None = None,
        random_state: int = 42,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if block_size_km is not None and block_size_km <= 0:
            raise ValueError(f"block_size_km must be positive, got {block_size_km}")
        self.n_splits = n_splits
        self.block_size_km = block_size_km
        self.random_state = random_state

    def _auto_block_size_km(self, lats: np.ndarray, lons: np.ndarray) -> float:
        """Default: bbox diagonal / (n_splits × 2)."""
        lat_span = float(lats.max() - lats.min())
        lon_span = float(lons.max() - lons.min())
        center_lat = float(lats.mean())
        lat_span_km = lat_span * 111.0
        lon_span_km = lon_span * 111.0 * max(math.cos(math.radians(center_lat)), 1e-6)
        diag_km = math.sqrt(lat_span_km**2 + lon_span_km**2)
        return diag_km / (self.n_splits * 2)

    def split(
        self,
        df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) для каждого фолда."""
        if lat_col not in df.columns or lon_col not in df.columns:
            raise KeyError(f"lat_col {lat_col!r} or lon_col {lon_col!r} not in dataframe")

        n = len(df)
        if n == 0:
            raise ValueError("Empty dataframe")

        lats = df[lat_col].to_numpy(dtype=float)
        lons = df[lon_col].to_numpy(dtype=float)

        bs_km = self.block_size_km or self._auto_block_size_km(lats, lons)
        center_lat = float(lats.mean())
        deg_lat, deg_lon = _km_to_deg(center_lat, bs_km)

        i = ((lats - lats.min()) / deg_lat).astype(int)
        j = ((lons - lons.min()) / deg_lon).astype(int)
        block_ids = np.array([f"{a}_{b}" for a, b in zip(i, j, strict=True)])
        unique_blocks = np.unique(block_ids)

        if len(unique_blocks) < self.n_splits:
            raise ValueError(
                f"Need at least {self.n_splits} unique blocks, "
                f"got {len(unique_blocks)} with block_size_km={bs_km:.2f}. "
                "Try smaller block_size_km."
            )

        rng = np.random.default_rng(self.random_state)
        shuffled = unique_blocks.copy()
        rng.shuffle(shuffled)
        folds_blocks = np.array_split(shuffled, self.n_splits)

        indices = np.arange(n)
        for fold_blocks in folds_blocks:
            test_mask = np.isin(block_ids, fold_blocks)
            test_idx = indices[test_mask]
            train_idx = indices[~test_mask]
            yield train_idx, test_idx

    def get_n_splits(
        self,
        df: pd.DataFrame | None = None,
        lat_col: str | None = None,
        lon_col: str | None = None,
    ) -> int:
        """Returns the number of splits (sklearn-compatible signature)."""
        return self.n_splits
