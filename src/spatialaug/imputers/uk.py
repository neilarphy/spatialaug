"""Universal Kriging imputer — regional_linear drift and external drift (KED).

In contrast to KrigingImputer (OK with an OLS coordinate-space
detrend applied before kriging), UK embeds the trend term directly
into the kriging system via pykrige.uk.UniversalKriging, which
solves trend and variogram jointly (GLS-style). The two are closely
related but not numerically identical when residuals are spatially
autocorrelated — comparing them on the benchmark is therefore
meaningful, not redundant.

When feature_cols are passed, UK switches to KED mode (Kriging with
External Drift): those features enter the kriging system as drift
terms and kriging models the residuals only. This puts kriging on
roughly the same input footing as a feature-based ML model
(GBM_full / KpR): same covariates, different functional form (GLS +
variogram vs gradient-boosted trees).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pykrige.uk import UniversalKriging

from spatialaug.imputers.base import Imputer

_VALID_VARIOGRAM_MODELS = ("spherical", "exponential", "gaussian", "linear", "power")
_VALID_DRIFT_TERMS = ("regional_linear", "specified")


class UniversalKrigingImputer(Imputer):
    """Universal Kriging imputer.

    Two modes:
    1. Regional linear (default) — built-in trend β0 + β1·lat + β2·lon.
    2. External drift / KED — engaged when feature_cols is given. The
       listed features must be known in both the training and query
       points; kriging uses them as the detrend component.

    Parameters
    ----------
    variogram_model : str, default="spherical"
        Variogram model.
    drift_terms : tuple of str, default=("regional_linear",)
        Ignored when feature_cols is set — in that case ("specified",)
        is used automatically.
    feature_cols : list or tuple of str, optional
        Feature columns used as external drift. Must be non-null in
        both training and query rows (KED does not tolerate missing
        covariates).
    standardize_features : bool, default=True
        Z-normalize drift features before passing them to kriging.
        Without this, features with very different scales (rooms vs
        area_in_m2) can wreck the numerical stability of the kriging
        matrices.
    log_transform : bool or "auto", default="auto"
        Log-transform the target before fitting.
    log_skew_threshold : float, default=1.0
        Threshold used when log_transform="auto": apply log when
        |skew| > threshold.
    dedupe_coords : bool, default=True
        Average target (and drift features) over rows sharing
        coordinates rounded to coord_round_decimals. Required to keep
        the kriging matrix non-singular.
    coord_round_decimals : int, default=4
        Rounding precision used by dedupe_coords.
    max_train_points : int or None, default=2000
        Random subsample of the training set used for fitting. Caps
        memory and time on large cities. None disables subsampling.
    random_state : int, default=42
        Seed for the random subsample.

    Examples
    --------
    Plain UK (coordinate-only trend):
        >>> imp = UniversalKrigingImputer()
        >>> df_filled = imp.fit_transform(df, lat="geo_lat", lon="geo_lon", target="price")

    KED with features:
        >>> imp = UniversalKrigingImputer(feature_cols=["area", "rooms"])
        >>> df_filled = imp.fit_transform(df, lat="geo_lat", lon="geo_lon", target="price")
    """

    def __init__(
        self,
        variogram_model: str = "spherical",
        drift_terms: tuple[str, ...] = ("regional_linear",),
        feature_cols: list[str] | tuple[str, ...] | None = None,
        standardize_features: bool = True,
        log_transform: bool | str = "auto",
        log_skew_threshold: float = 1.0,
        dedupe_coords: bool = True,
        coord_round_decimals: int = 4,
        max_train_points: int | None = 2000,
        random_state: int = 42,
    ) -> None:
        
        super().__init__()

        if variogram_model not in _VALID_VARIOGRAM_MODELS:
            raise ValueError(
                f"variogram_model must be one of {_VALID_VARIOGRAM_MODELS}, "
                f"got {variogram_model!r}"
            )
        
        if feature_cols is not None:
            drift_terms = ("specified",)
            feature_cols = tuple(feature_cols)
            if not feature_cols:
                raise ValueError("feature_cols must contain at least one column")
            
        invalid_drifts = set(drift_terms) - set(_VALID_DRIFT_TERMS)

        if invalid_drifts:
            raise ValueError(
                f"Unsupported drift_terms: {invalid_drifts}. "
                f"Supported: {_VALID_DRIFT_TERMS}"
            )
        
        if log_transform not in (True, False, "auto"):
            raise ValueError(f"log_transform must be bool or 'auto', got {log_transform!r}")
        
        if max_train_points is not None and max_train_points < 5:
            raise ValueError(f"max_train_points must be >= 5 or None, got {max_train_points}")
        
        self.variogram_model = variogram_model
        self.drift_terms = tuple(drift_terms)
        self.feature_cols = feature_cols
        self.standardize_features = standardize_features
        self.log_transform = log_transform
        self.log_skew_threshold = log_skew_threshold
        self.dedupe_coords = dedupe_coords
        self.coord_round_decimals = coord_round_decimals
        self.max_train_points = max_train_points
        self.random_state = random_state

        self.log_applied_: bool | None = None
        self.kriging_: UniversalKriging | None = None
        self.feature_means_: np.ndarray | None = None
        self.feature_stds_: np.ndarray | None = None

    def fit(
        self,
        df: pd.DataFrame,
        lat: str,
        lon: str,
        target: str,
        **kwargs,
    ) -> UniversalKrigingImputer:
        self.lat_col = lat
        self.lon_col = lon
        self.target_col = target

        if self.feature_cols is not None:
            missing = [c for c in self.feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"feature_cols missing from df: {missing}")

        observed = df[df[target].notna()]
        if self.feature_cols is not None:
            observed = observed.copy()
            
            for col in self.feature_cols:
                if not pd.api.types.is_numeric_dtype(observed[col]):
                    observed[col] = pd.to_numeric(observed[col], errors="coerce")
            
            feat_mask = observed[list(self.feature_cols)].notna().all(axis=1)
            observed = observed[feat_mask]

        if observed.empty:
            raise ValueError(
                f"No observed rows in target {target!r}"
                + (" with all feature_cols non-null" if self.feature_cols else "")
            )

        if self.dedupe_coords:
            feat_list = list(self.feature_cols) if self.feature_cols else []
            agg_dict = {lat: "first", lon: "first", target: "mean"}
            for c in feat_list:
                agg_dict[c] = "mean"
            cols_to_keep = [lat, lon, target] + feat_list
            observed = (
                observed[cols_to_keep]
                .assign(
                    _lat_r=lambda d: d[lat].round(self.coord_round_decimals),
                    _lon_r=lambda d: d[lon].round(self.coord_round_decimals),
                )
                .groupby(["_lat_r", "_lon_r"], as_index=False)
                .agg(agg_dict)
                [cols_to_keep]
            )

        if self.max_train_points is not None and len(observed) > self.max_train_points:
            observed = observed.sample(
                n=self.max_train_points, random_state=self.random_state
            )

        if len(observed) < 5:
            raise ValueError(
                f"Need at least 5 observed (deduped) points, got {len(observed)}"
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
                    "log_transform requires positive values; found non-positive entries"
                )
            y = np.log(y)

        kriging_kwargs = {"drift_terms": list(self.drift_terms)}
        if self.feature_cols is not None:
            features = observed[list(self.feature_cols)].to_numpy(dtype=float)
            if self.standardize_features:
                self.feature_means_ = features.mean(axis=0)
                self.feature_stds_ = features.std(axis=0)
                self.feature_stds_ = np.where(
                    self.feature_stds_ < 1e-9, 1.0, self.feature_stds_
                )
                features = (features - self.feature_means_) / self.feature_stds_
            kriging_kwargs["specified_drift"] = [
                features[:, i] for i in range(features.shape[1])
            ]

        self.kriging_ = UniversalKriging(
            coords[:, 1],
            coords[:, 0],
            y,
            variogram_model=self.variogram_model,
            verbose=False,
            enable_plotting=False,
            pseudo_inv=True,
            **kriging_kwargs,
        )
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        result = df.copy()
        mask = result[self.target_col].isna()
        if not mask.any():
            return result

        query_coords = result.loc[mask, [self.lat_col, self.lon_col]].to_numpy(dtype=float)
        execute_kwargs = {}
        
        if self.feature_cols is not None:
            query_df = result.loc[mask, list(self.feature_cols)].copy()
            for col in self.feature_cols:
                if not pd.api.types.is_numeric_dtype(query_df[col]):
                    query_df[col] = pd.to_numeric(query_df[col], errors="coerce")
            query_features = query_df.to_numpy(dtype=float)
            if np.isnan(query_features).any():
                raise ValueError(
                    "KED imputer received NaN in feature_cols at query points. "
                    "External drift kriging requires covariates known at all query locations."
                )
            if self.standardize_features:
                query_features = (
                    (query_features - self.feature_means_) / self.feature_stds_
                )
            execute_kwargs["specified_drift_arrays"] = [
                query_features[:, i] for i in range(query_features.shape[1])
            ]

        z, ss = self.kriging_.execute(
            "points", query_coords[:, 1], query_coords[:, 0], **execute_kwargs
        )
        predictions = np.asarray(z, dtype=float)
        variances = np.asarray(ss, dtype=float)

        if self.log_applied_:
            if self.feature_cols is not None:
                predictions = np.exp(np.clip(predictions, -20, 25))
            else:
                predictions = np.exp(
                    np.clip(predictions + 0.5 * variances, -20, 25),
                )

        result.loc[mask, self.target_col] = predictions
        return result
