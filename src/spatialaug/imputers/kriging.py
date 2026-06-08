from __future__ import annotations

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from scipy.spatial import KDTree
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from spatialaug.imputers.base import Imputer

_VALID_VARIOGRAM_MODELS = ("spherical", "exponential", "gaussian", "linear", "power")
_VALID_DETREND = (None, "linear", "quadratic")


class KrigingImputer(Imputer):
    """Ordinary Kriging imputer for spatial data.

    Parameters
    ----------
    variogram_model : str, default="spherical"
        Variogram model: "spherical", "exponential", "gaussian",
        "linear" or "power".
    log_transform : bool or "auto", default="auto"
        Apply log(target) before fitting. "auto" triggers it when
        |skewness| > log_skew_threshold. Back-transform uses the
        lognormal correction exp(pred + 0.5 * variance).
    detrend : {None, "linear", "quadratic"}, default="linear"
        Optional polynomial detrending in coordinate space before
        kriging. The trend is subtracted from y, kriging models the
        residuals, and the trend is added back in transform.
    local : bool, default=False
        If True, refit a small kriging model on the n_neighbors closest
        training points for each query (slower but scales to large
        training sets and adapts to local non-stationarity).
    n_neighbors : int, default=500
        Number of neighbors used per query in local mode.
    log_skew_threshold : float, default=1.0
        Skewness threshold for log_transform="auto".
    dedupe_coords : bool, default=True
        Average target over rows sharing coordinates (rounded to
        coord_round_decimals) before fitting. Required: duplicates
        make the kriging matrix singular.
    coord_round_decimals : int, default=4
        Rounding precision used by dedupe_coords.
    progress : bool, default=False
        Show a tqdm bar in local kriging mode.

    Examples
    --------
    >>> imp = KrigingImputer(variogram_model="spherical", log_transform="auto")
    >>> df_filled = imp.fit_transform(df, lat="geo_lat", lon="geo_lon", target="price")
    """

    def __init__(
        self,
        variogram_model: str = "spherical",
        log_transform: bool | str = "auto",
        detrend: str | None = "linear",
        local: bool = False,
        n_neighbors: int = 500,
        log_skew_threshold: float = 1.0,
        dedupe_coords: bool = True,
        coord_round_decimals: int = 4,
        progress: bool = False,
    ) -> None:
        
        super().__init__()
        
        if variogram_model not in _VALID_VARIOGRAM_MODELS:
            raise ValueError(
                f"variogram_model must be one of {_VALID_VARIOGRAM_MODELS}, got {variogram_model!r}"
            )
        
        if detrend not in _VALID_DETREND:
            raise ValueError(f"detrend must be None / 'linear' / 'quadratic', got {detrend!r}")
        
        if log_transform not in (True, False, "auto"):
            raise ValueError(f"log_transform must be bool or 'auto', got {log_transform!r}")
        
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}")

        self.variogram_model = variogram_model
        self.log_transform = log_transform
        self.detrend = detrend
        self.local = local
        self.n_neighbors = n_neighbors
        self.log_skew_threshold = log_skew_threshold
        self.dedupe_coords = dedupe_coords
        self.coord_round_decimals = coord_round_decimals
        self.progress = progress

        self.log_applied_: bool | None = None
        self.trend_model_: LinearRegression | None = None
        self.coords_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None
        self.kriging_: OrdinaryKriging | None = None
        self.tree_: KDTree | None = None

    def _trend_features(self, coords: np.ndarray) -> np.ndarray:
        """Build feature matrix for trend regression based on self.detrend."""
        
        if self.detrend == "linear":
            return coords
        
        return np.column_stack([coords, coords**2, coords[:, 0] * coords[:, 1]])

    def fit(
        self,
        df: pd.DataFrame,
        lat: str,
        lon: str,
        target: str,
        **kwargs,
    ) -> KrigingImputer:
        self.lat_col = lat
        self.lon_col = lon
        self.target_col = target

        observed = df[df[target].notna()]
        if observed.empty:
            raise ValueError(f"No observed (non-NaN) rows in target {target!r}")

        if self.dedupe_coords:
            observed = (
                observed[[lat, lon, target]]
                .assign(
                    _lat_r=lambda d: d[lat].round(self.coord_round_decimals),
                    _lon_r=lambda d: d[lon].round(self.coord_round_decimals),
                )
                .groupby(["_lat_r", "_lon_r"], as_index=False)
                .agg({lat: "first", lon: "first", target: "mean"})
                [[lat, lon, target]]
            )

        if len(observed) < 5:
            raise ValueError(
                f"Need at least 5 observed (deduped) points for variogram fitting, "
                f"got {len(observed)}"
            )

        coords = observed[[lat, lon]].to_numpy(dtype=float)
        y = observed[target].to_numpy(dtype=float)

        if self.log_transform == "auto":
            skew = float(pd.Series(y).skew())
            self.log_applied_ = abs(skew) > self.log_skew_threshold
        else:
            self.log_applied_ = bool(self.log_transform)

        if self.log_applied_:
            if (y <= 0).any():
                raise ValueError(
                    "log_transform requires positive values; "
                    "found non-positive entries. Filter before fit or set "
                    "log_transform=False."
                )
            y = np.log(y)

        if self.detrend is not None:
            features = self._trend_features(coords)
            self.trend_model_ = LinearRegression().fit(features, y)
            residuals = y - self.trend_model_.predict(features)
        else:
            self.trend_model_ = None
            residuals = y

        self.coords_ = coords
        self.residuals_ = residuals

        if not self.local:
            self.kriging_ = OrdinaryKriging(
                coords[:, 1], 
                coords[:, 0], 
                residuals,
                variogram_model=self.variogram_model,
                verbose=False,
                enable_plotting=False,
                pseudo_inv=True,  
            )
        else:
            self.tree_ = KDTree(coords)

        self.is_fitted = True

        return self

    def _kriging_predict(self, query_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (predictions, variances) for query points in residual space."""
        
        query_lat = query_coords[:, 0]
        query_lon = query_coords[:, 1]

        if not self.local:
            z, ss = self.kriging_.execute("points", query_lon, query_lat)
            return np.asarray(z, dtype=float), np.asarray(ss, dtype=float)

        n = len(query_coords)
        predictions = np.empty(n, dtype=float)
        variances = np.empty(n, dtype=float)
        k = min(self.n_neighbors, len(self.coords_))

        iterator = (
            tqdm(range(n), desc="local kriging", leave=False)
            if self.progress
            else range(n)
        )
        for i in iterator:
            _, idx = self.tree_.query(query_coords[i : i + 1], k=k)
            idx = np.atleast_1d(idx.ravel())
            local_coords = self.coords_[idx]
            local_residuals = self.residuals_[idx]
            ok = OrdinaryKriging(
                local_coords[:, 1],
                local_coords[:, 0],
                local_residuals,
                variogram_model=self.variogram_model,
                verbose=False,
                enable_plotting=False,
                pseudo_inv=True,
            )
            z, ss = ok.execute("points", [query_lon[i]], [query_lat[i]])
            predictions[i] = float(np.asarray(z).item())
            variances[i] = float(np.asarray(ss).item())

        return predictions, variances

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        
        self._check_fitted()
        result = df.copy()
        mask = result[self.target_col].isna()
        
        if not mask.any():
            return result

        query_coords = result.loc[mask, [self.lat_col, self.lon_col]].to_numpy(dtype=float)

        predictions, variances = self._kriging_predict(query_coords)

        if self.trend_model_ is not None:
            trend = self.trend_model_.predict(self._trend_features(query_coords))
            predictions = predictions + trend

        if self.log_applied_:
            predictions = np.exp(predictions + 0.5 * variances)

        result.loc[mask, self.target_col] = predictions
        
        return result
