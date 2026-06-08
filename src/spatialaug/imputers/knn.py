from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from spatialaug.imputers.base import Imputer


class KNNImputer(Imputer):
    """Geographic k-Nearest Neighbors imputer.

    Fills missing values with the mean (or median) of the k nearest
    observed points. Unlike IDW, all neighbors are weighted equally.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors to aggregate over.
    aggregation : {"mean", "median"}, default="mean"
        How to aggregate the neighbor values.

    Examples
    --------
    >>> imp = KNNImputer(n_neighbors=5, aggregation="median")
    >>> df_filled = imp.fit_transform(df, lat="geo_lat", lon="geo_lon", target="price")
    """

    def __init__(self, n_neighbors: int = 5, aggregation: str = "mean") -> None:
        super().__init__()

        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")
        if aggregation not in ("mean", "median"):
            raise ValueError(f"aggregation must be 'mean' or 'median', got {aggregation!r}")
        
        self.n_neighbors = n_neighbors
        self.aggregation = aggregation
        self.tree_: KDTree | None = None
        self.values_: np.ndarray | None = None

    def fit(
        self,
        df: pd.DataFrame,
        lat: str,
        lon: str,
        target: str,
        **kwargs,
    ) -> KNNImputer:
        
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
        _, indices = self.tree_.query(query_coords, k=self.n_neighbors)
        indices = np.asarray(indices)

        if self.n_neighbors == 1:
            indices = indices[:, None]

        neighbor_values = self.values_[indices]

        if self.aggregation == "mean":
            predictions = neighbor_values.mean(axis=1)
        else:
            predictions = np.median(neighbor_values, axis=1)

        result.loc[mask, self.target_col] = predictions

        return result
