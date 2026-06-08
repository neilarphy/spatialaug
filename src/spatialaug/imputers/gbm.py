"""LightGBM imputer with spatial features.

ML baseline for the benchmark. Takes lat/lon plus optional
feature_cols as inputs and trains a regressor to predict target.

Auto log-transform for skewed targets (same logic as KrigingImputer).
No bias correction is applied on inversion because GBM predicts
directly in log space and plain exp() recovers a median estimate.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from spatialaug.imputers.base import Imputer


class GBMImputer(Imputer):
    """LightGBM imputer using lat/lon plus optional extra features.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting trees.
    learning_rate : float, default=0.1
        Learning rate.
    num_leaves : int, default=31
        Maximum number of leaves per tree.
    feature_cols : list[str], optional
        Extra columns appended to lat/lon as features.
    log_transform : bool or "auto", default="auto"
        Apply log(target) before fitting. "auto" triggers it when
        |skewness| > log_skew_threshold.
    log_skew_threshold : float, default=1.0
        Skewness threshold for log_transform="auto".
    random_state : int, default=42
        Seed for LightGBM.

    Examples
    --------
    >>> imp = GBMImputer(feature_cols=["area", "rooms"])
    >>> df_filled = imp.fit_transform(df, lat="geo_lat", lon="geo_lon", target="price")
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        feature_cols: list[str] | None = None,
        log_transform: bool | str = "auto",
        log_skew_threshold: float = 1.0,
        random_state: int = 42,
    ) -> None:
        
        super().__init__()
    
        if n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {n_estimators}")
    
        if log_transform not in (True, False, "auto"):
            raise ValueError(f"log_transform must be bool or 'auto', got {log_transform!r}")
    
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.feature_cols = list(feature_cols) if feature_cols else None
        self.log_transform = log_transform
        self.log_skew_threshold = log_skew_threshold
        self.random_state = random_state

        self.log_applied_: bool | None = None
        self.model_: lgb.LGBMRegressor | None = None
        self._used_features_: list[str] | None = None

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [self.lat_col, self.lon_col]
    
        if self.feature_cols:
            cols = cols + self.feature_cols
    
        self._used_features_ = cols
    
        return df[cols].astype(float)

    def fit(
        self,
        df: pd.DataFrame,
        lat: str,
        lon: str,
        target: str,
        **kwargs,
    ) -> GBMImputer:
        
        self.lat_col = lat
        self.lon_col = lon
        self.target_col = target

        if self.feature_cols:
            missing_cols = [c for c in self.feature_cols if c not in df.columns]
            if missing_cols:
                raise KeyError(f"feature_cols not in dataframe: {missing_cols}")

        observed = df[df[target].notna()]
       
        if observed.empty:
            raise ValueError(f"No observed rows in target {target!r}")
       
        if len(observed) < 10:
            raise ValueError(f"Need at least 10 observed points for GBM, got {len(observed)}")

        X = self._build_features(observed)
        y = observed[target].to_numpy(dtype=float)

        if self.log_transform == "auto":
            skew = float(pd.Series(y).skew())
            self.log_applied_ = abs(skew) > self.log_skew_threshold
        else:
            self.log_applied_ = bool(self.log_transform)

        if self.log_applied_:
            if (y <= 0).any():
                raise ValueError(
                    "log_transform requires positive values; found non-positive entries"
                )
            y = np.log(y)

        self.model_ = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            verbosity=-1,
        )
        self.model_.fit(X, y)
        self.is_fitted = True
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        result = df.copy()
        mask = result[self.target_col].isna()
    
        if not mask.any():
            return result

        X_query = self._build_features(result.loc[mask])
        predictions = self.model_.predict(X_query)

        if self.log_applied_:
            predictions = np.exp(predictions)

        result.loc[mask, self.target_col] = predictions
    
        return result
