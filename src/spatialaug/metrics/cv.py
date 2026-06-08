from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd

from spatialaug.utils.geo import KM_PER_DEGREE, km_to_deg


class SpatialBlockCV:
    """K-fold cross-validator with spatial-block splitting.

    Plain random K-fold underestimates generalization error on
    spatial data because neighboring points are autocorrelated:
    held-out rows have near-duplicates in the training fold. This
    splitter partitions the bounding box into square blocks and
    assigns whole blocks (not individual rows) to folds, removing
    that leak.

    Optionally supports buffered CV: 
    rows in a buffer zone of buffer_km around any test row
    are excluded from training to also remove short-range
    autocorrelation leakage. Recommended buffer width is roughly the
    variogram range of the target.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    block_size_km : float, optional
        Block size in km. If None, computed from the data bounding
        box as bbox_diagonal / (n_splits * 2). The auto value is a
        heuristic — for honest reporting prefer to set it to roughly
        the variogram range of the target.
    buffer_km : float, default=0.0
        Width of the buffer zone between test and train blocks.
        Training rows within buffer_km of any test row are dropped.
        0 disables the buffer (classical Spatial Block CV).
    random_state : int, default=42
        Seed for the block shuffle across folds.

    Examples
    --------
    >>> cv = SpatialBlockCV(n_splits=5, block_size_km=10)
    >>> for train_idx, test_idx in cv.split(df, lat_col="lat", lon_col="lon"):
    ...     X_train = df.iloc[train_idx]
    ...     X_test = df.iloc[test_idx]
    >>>
    >>> # Buffered version — recommended for kriging-style methods
    >>> cv_buf = SpatialBlockCV(n_splits=5, block_size_km=10, buffer_km=5)
    """

    def __init__(
        self,
        n_splits: int = 5,
        block_size_km: float | None = None,
        buffer_km: float = 0.0,
        random_state: int = 42,
    ) -> None:
        
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        
        if block_size_km is not None and block_size_km <= 0:
            raise ValueError(f"block_size_km must be positive, got {block_size_km}")
        
        if buffer_km < 0:
            raise ValueError(f"buffer_km must be >= 0, got {buffer_km}")
        
        self.n_splits = n_splits
        self.block_size_km = block_size_km
        self.buffer_km = buffer_km
        self.random_state = random_state

    def _auto_block_size_km(self, lats: np.ndarray, lons: np.ndarray) -> float:
        """Default: bbox diagonal / (n_splits × 2)."""
        
        lat_span = float(lats.max() - lats.min())
        lon_span = float(lons.max() - lons.min())
        
        center_lat = float(lats.mean())
        
        lat_span_km = lat_span * KM_PER_DEGREE
        lon_span_km = lon_span * KM_PER_DEGREE * max(math.cos(math.radians(center_lat)), 1e-6)
        
        diag_km = math.sqrt(lat_span_km**2 + lon_span_km**2)
        
        return diag_km / (self.n_splits * 2)

    def split(
        self,
        df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) for each fold."""

        if lat_col not in df.columns or lon_col not in df.columns:
            raise KeyError(f"lat_col {lat_col!r} or lon_col {lon_col!r} not in dataframe")

        n = len(df)
        
        if n == 0:
            raise ValueError("Empty dataframe")

        lats = df[lat_col].to_numpy(dtype=float)
        lons = df[lon_col].to_numpy(dtype=float)

        bs_km = self.block_size_km or self._auto_block_size_km(lats, lons)
        center_lat = float(lats.mean())
        deg_lat, deg_lon = km_to_deg(center_lat, bs_km)

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
            train_mask = ~test_mask

            if self.buffer_km > 0 and len(test_idx) > 0:
                test_lats = lats[test_idx]
                test_lons = lons[test_idx]
                lat_diff_km = (
                    (lats[train_mask][:, None] - test_lats[None, :]) * KM_PER_DEGREE
                )
                lon_diff_km = (
                    (lons[train_mask][:, None] - test_lons[None, :])
                    * KM_PER_DEGREE * math.cos(math.radians(center_lat))
                )
                dist_km = np.sqrt(lat_diff_km ** 2 + lon_diff_km ** 2)
                min_dist_to_test = dist_km.min(axis=1)
                train_idx_full = indices[train_mask]
                buffer_keep = min_dist_to_test > self.buffer_km
                train_idx = train_idx_full[buffer_keep]
            else:
                train_idx = indices[train_mask]
            
            yield train_idx, test_idx

    def get_n_splits(
        self,
        df: pd.DataFrame | None = None,
        lat_col: str | None = None,
        lon_col: str | None = None,
    ) -> int:
        """Returns the number of splits (sklearn-compatible signature)."""
        
        return self.n_splits
