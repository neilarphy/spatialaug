from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from spatialaug.imputers.base import Imputer


class IDWImputer(Imputer):
    """Inverse Distance Weighted imputer.

    Fills missing values with a weighted average of the k nearest
    observed points. Each neighbor's weight is proportional to
    1 / distance ** power.

    Parameters
    ----------
    power : float, default=2
        Distance exponent. Higher values give more weight to the
        closest neighbors. Typical range: 1-3.
    n_neighbors : int, default=10
        Number of nearest neighbors to average over.

    Examples
    --------
    >>> imp = IDWImputer(power=2, n_neighbors=10)
    >>> df_filled = imp.fit_transform(df, lat="geo_lat", lon="geo_lon", target="price")
    """

    def __init__(self, power: float = 2.0, n_neighbors: int = 10) -> None:
        super().__init__()
       
        if power <= 0:
            raise ValueError(f"power must be positive, got {power}")
       
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
       
        self.power = power
        self.n_neighbors = n_neighbors
        self.tree_: KDTree | None = None
        self.values_: np.ndarray | None = None

    def fit(
        self,
        df: pd.DataFrame,
        lat: str,
        lon: str,
        target: str,
        **kwargs,
    ) -> IDWImputer:
        
        self.lat_col = lat
        self.lon_col = lon
        self.target_col = target

        observed = df[df[target].notna()]
      
        if observed.empty:
            raise ValueError(f"No observed (non-NaN) rows in target column {target!r}")
      
        if len(observed) < self.n_neighbors:
            raise ValueError(
                f"Need at least {self.n_neighbors} observed points, got {len(observed)}"
            )

        coords = observed[[lat, lon]].to_numpy(dtype=float)
        self.tree_ = KDTree(coords)
        self.values_ = observed[target].to_numpy(dtype=float)
        self.is_fitted = True
      
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        result = df.copy()
        mask = result[self.target_col].isna()
        
        if not mask.any():
            return result

        query_coords = result.loc[mask, [self.lat_col, self.lon_col]].to_numpy(dtype=float)
        distances, indices = self.tree_.query(query_coords, k=self.n_neighbors)
        distances = np.asarray(distances)
        indices = np.asarray(indices)

        if self.n_neighbors == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        eps = 1e-12
        weights = 1.0 / np.maximum(distances, eps) ** self.power
        weights /= weights.sum(axis=1, keepdims=True)
        predictions = (weights * self.values_[indices]).sum(axis=1)

        result.loc[mask, self.target_col] = predictions
        
        return result
