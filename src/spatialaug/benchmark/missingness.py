"""Missingness mechanism simulators for the benchmark.

Three mechanisms:
- MCAR (Missing Completely At Random): uniformly random rows.
- diffuse MNAR: whole spatial clusters (mimics under-collection of
  entire zones, e.g. coverage gaps in commercial data sources).
- focused MNAR: rows with extreme target values (mimics value-based
  privacy redaction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

_VALID_MECHANISMS = ("mcar", "diffuse_mnar", "focused_mnar")


class MissingnessSimulator:
    """Synthetic missingness generator.

    Parameters
    ----------
    mechanism : {"mcar", "diffuse_mnar", "focused_mnar"}
        Which missingness mechanism to simulate.
    ratio : float, default=0.3
        Fraction of observable rows to mask (in (0, 1)).
    random_state : int, default=42
        Seed for all internal randomness.
    n_clusters : int, default=5
        Number of cluster centres used by diffuse_mnar.
    tail : {"high", "low"}, default="high"
        Which tail focused_mnar targets: "high" (large target values)
        or "low" (small values).
    focused_pool_factor : float, default=2.0
        Pool size for the focused_mnar random sampling relative to
        n_to_mask. 1.0 deterministically takes the top-N% by value
        (no seed variability). 2.0 takes the top-(2 * ratio)% and
        then randomly subsamples ratio% from that pool, so the seed
        affects which extreme rows are masked but the bias toward
        large values is preserved. The latter is closer to how real
        privacy filters behave — they redact extremes probabilistically,
        not all top values at once.

    Examples
    --------
    >>> sim = MissingnessSimulator(mechanism="focused_mnar", ratio=0.1)
    >>> df_masked, mask = sim.apply(df, target_col="price")
    """

    def __init__(
        self,
        mechanism: str = "mcar",
        ratio: float = 0.3,
        random_state: int = 42,
        n_clusters: int = 5,
        tail: str = "high",
        focused_pool_factor: float = 2.0,
    ) -> None:
        
        if mechanism not in _VALID_MECHANISMS:
            raise ValueError(
                f"mechanism must be one of {_VALID_MECHANISMS}, got {mechanism!r}"
            )
        
        if not (0 < ratio < 1):
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        
        if tail not in ("high", "low"):
            raise ValueError(f"tail must be 'high' or 'low', got {tail!r}")
        
        if focused_pool_factor < 1.0:
            raise ValueError(
                f"focused_pool_factor must be >= 1.0, got {focused_pool_factor}"
            )
        
        self.mechanism = mechanism
        self.ratio = ratio
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.tail = tail
        self.focused_pool_factor = focused_pool_factor

    def apply(
        self,
        df: pd.DataFrame,
        target_col: str,
        lat_col: str | None = None,
        lon_col: str | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Apply the configured mechanism to df.

        Returns
        -------
        (masked_df, mask) : tuple
            masked_df — a copy of df with target_col set to NaN on the
            masked rows.
            mask — a boolean array of length len(df). True marks the
            rows that were masked.
        """
        
        if target_col not in df.columns:
            raise KeyError(f"target_col {target_col!r} not in dataframe")

        observable = df[target_col].notna().to_numpy()
        observable_idx = np.where(observable)[0]
        n_observable = len(observable_idx)
        if n_observable == 0:
            raise ValueError("All values in target_col are already NaN")

        n_to_mask = int(n_observable * self.ratio)
        if n_to_mask < 1:
            raise ValueError(
                f"ratio={self.ratio} gives <1 point to mask out of {n_observable}"
            )

        if self.mechanism == "mcar":
            to_mask = self._mcar(observable_idx, n_to_mask)
        elif self.mechanism == "diffuse_mnar":
            if lat_col is None or lon_col is None:
                raise ValueError("diffuse_mnar requires lat_col and lon_col")
            to_mask = self._diffuse_mnar(df, observable_idx, n_to_mask, lat_col, lon_col)
        else:  # focused_mnar
            to_mask = self._focused_mnar(df, observable_idx, n_to_mask, target_col)

        result = df.copy()
        result.iloc[to_mask, result.columns.get_loc(target_col)] = np.nan

        mask = np.zeros(len(df), dtype=bool)
        mask[to_mask] = True
        return result, mask

    def _mcar(self, observable_idx: np.ndarray, n_to_mask: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)

        return rng.choice(observable_idx, size=n_to_mask, replace=False)

    def _diffuse_mnar(
        self,
        df: pd.DataFrame,
        observable_idx: np.ndarray,
        n_to_mask: int,
        lat_col: str,
        lon_col: str,
    ) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        coords = df.iloc[observable_idx][[lat_col, lon_col]].to_numpy(dtype=float)

        n_centers = min(self.n_clusters, len(coords))
        center_idx = rng.choice(len(coords), size=n_centers, replace=False)
        centers = coords[center_idx]

        tree = KDTree(centers)
        dists, _ = tree.query(coords, k=1)

        order = np.argsort(dists)

        return observable_idx[order[:n_to_mask]]

    def _focused_mnar(
        self,
        df: pd.DataFrame,
        observable_idx: np.ndarray,
        n_to_mask: int,
        target_col: str,
    ) -> np.ndarray:
        values = df.iloc[observable_idx][target_col].to_numpy(dtype=float)
        
        if self.tail == "high":
            order = np.argsort(values)[::-1]  
        else:
            order = np.argsort(values) 

        pool_size = min(
            int(n_to_mask * self.focused_pool_factor), len(observable_idx)
        )
        pool = observable_idx[order[:pool_size]]
        
        if pool_size == n_to_mask:
            return pool
        
        rng = np.random.default_rng(self.random_state)
        
        return rng.choice(pool, size=n_to_mask, replace=False)
