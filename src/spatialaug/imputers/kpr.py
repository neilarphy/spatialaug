"""KpR — Kriging prior Regression (geostat x ML hybrid).

Pipeline:
1. Prior step: Ordinary Kriging is fit on observed rows and predicts
   the target at every row.
2. Refine step: a GBM (LightGBM) is trained with the kriging
   prediction as an extra feature alongside any other observed
   features.

Pure kriging captures spatial smoothness; pure GBM captures
non-spatial feature signal. KpR combines both.

Out-of-fold prior
-----------------
At fit time the kriging prior at each training row must come from a
kriging model that has not seen that row's true y. Otherwise the
prior is suspiciously close to the truth (with local=False, OK at
point x_i puts non-zero weight on its own y_i), GBM trusts it too
much, and inference metrics — where the prior is a genuine estimate
— drop relative to train.

Default behaviour (oof_folds=5): split observed rows into k folds,
fit kriging on (k-1) folds and predict the held-out fold. Cost is k
extra kriging fits per KpR.fit(). Pass cv=SpatialBlockCV(...) for
spatial-aware folds (recommended for honest reporting on spatial
data, since random KFold lets neighbouring autocorrelated points
leak across train/val).

Setting oof_folds=None falls back to the leaky in-sample prior — only
useful for reproducing pre-OOF benchmark numbers; within-region
metrics will be biased upward.

Inference (transform) is leak-free by construction: kriging is fit
on observed train rows and only predicts NaN rows it has never seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from spatialaug.imputers.base import Imputer
from spatialaug.imputers.gbm import GBMImputer
from spatialaug.imputers.kriging import KrigingImputer
from spatialaug.metrics.cv import SpatialBlockCV

KRIGING_PRIOR_COL = "_kriging_prior"


class KrigingPriorRegression(Imputer):
    """KpR: kriging predictions used as an extra feature for GBM.

    Parameters
    ----------
    kriging_kwargs : dict, optional
        Parameters forwarded to the inner KrigingImputer. Default:
        {"variogram_model": "spherical", "log_transform": "auto",
         "detrend": "linear", "local": False, "progress": False}
    gbm_kwargs : dict, optional
        Parameters forwarded to the inner GBMImputer. Default:
        {"n_estimators": 100, "num_leaves": 31, "log_transform": "auto"}
    feature_cols : list[str], optional
        Additional observed features (the kriging prior is appended
        automatically as the final feature).
    oof_folds : int or None, default=5
        Number of folds for out-of-fold kriging prior computation at
        fit time. Set to None or 1 to disable OOF and use the leaky
        in-sample prior (only for benchmark reproducibility). See
        module docstring "Out-of-fold prior" section. Ignored when
        cv is provided (cv defines the splits).
    random_state : int, default=42
        Seed for the OOF KFold shuffle (ignored when cv is provided).
    cv : SpatialBlockCV or sklearn-style splitter or None, default=None
        Custom cross-validator used to generate OOF folds. When None,
        a random sklearn KFold(n_splits=oof_folds, shuffle=True) is
        used. Pass a SpatialBlockCV to get spatial-aware folds where
        neighbouring points cannot leak across train/val via spatial
        autocorrelation — recommended for honest reporting on spatial
        data.

    Examples
    --------
    Random OOF (default):
    >>> kpr = KrigingPriorRegression(feature_cols=["kkt_count", "is_mall"])
    >>> kpr.fit(df, lat="centroid_lat", lon="centroid_lon", target="avg_bill")
    >>> df_filled = kpr.transform(df)

    Spatial-block OOF (recommended for benchmark reporting):
    >>> from spatialaug.metrics import SpatialBlockCV
    >>> kpr = KrigingPriorRegression(
    ...     feature_cols=["kkt_count"],
    ...     cv=SpatialBlockCV(n_splits=5, buffer_km=2.0),
    ... )
    >>> kpr.fit(df, lat="centroid_lat", lon="centroid_lon", target="avg_bill")
    """

    def __init__(
        self,
        kriging_kwargs: dict | None = None,
        gbm_kwargs: dict | None = None,
        feature_cols: list[str] | None = None,
        oof_folds: int | None = 5,
        random_state: int = 42,
        cv: SpatialBlockCV | KFold | None = None,
    ) -> None:
        
        super().__init__()
        self.kriging_kwargs = kriging_kwargs or {
            "variogram_model": "spherical",
            "log_transform": "auto",
            "detrend": "linear",
            "local": False,
            "progress": False,
        }
        self.gbm_kwargs = gbm_kwargs or {
            "n_estimators": 100,
            "num_leaves": 31,
            "log_transform": "auto",
        }
        self.feature_cols = list(feature_cols) if feature_cols else []

        if oof_folds is not None and oof_folds < 2:
            oof_folds = None

        self.oof_folds = oof_folds
        self.random_state = int(random_state)
        self.cv = cv

        self.kriging_: KrigingImputer | None = None
        self.gbm_: GBMImputer | None = None

    def _compute_oof_prior(
        self,
        df_observed: pd.DataFrame,
        lat: str,
        lon: str,
        target: str,
        n_folds: int,
    ) -> np.ndarray:
        """Compute leak-free kriging prior at each observed row via k-fold.

        For each fold, kriging is fit on the other (k-1) folds and
        used to predict the held-out fold. The returned array aligns
        with df_observed row order. Uses self.cv if provided
        (SpatialBlockCV recommended), otherwise random sklearn KFold.
        """
        
        n = len(df_observed)
        oof = np.full(n, np.nan, dtype=float)

        if isinstance(self.cv, SpatialBlockCV):
            splitter = self.cv.split(df_observed, lat_col=lat, lon_col=lon)
        elif self.cv is not None:
            splitter = self.cv.split(df_observed)
        else:
            splitter = KFold(
                n_splits=n_folds, shuffle=True, random_state=self.random_state,
            ).split(df_observed)
        
        for train_idx, val_idx in splitter:
            train_fold = df_observed.iloc[train_idx].reset_index(drop=True)
            val_fold = df_observed.iloc[val_idx].copy().reset_index(drop=True)
            val_fold[target] = np.nan
            combined = pd.concat([train_fold, val_fold], ignore_index=True)
           
            try:
                krig = KrigingImputer(**self.kriging_kwargs)
                krig.fit(combined, lat=lat, lon=lon, target=target)
                filled = krig.transform(combined)
                preds = filled[target].to_numpy(dtype=float)[-len(val_fold):]
            except ValueError:
                preds = np.full(
                    len(val_fold), df_observed[target].mean(), dtype=float,
                )
            oof[val_idx] = preds

        if np.isnan(oof).any():
            oof[np.isnan(oof)] = float(df_observed[target].mean())

        return oof

    def _add_kriging_prior(
        self, df: pd.DataFrame, kriging_imputer: KrigingImputer,
    ) -> pd.DataFrame:
        """Attach a kriging-prediction column to every row of df.

        Used at inference (transform) where the kriging model was fit
        on a disjoint training set, so this is leak-free by construction.
        The kriging model was fit on observed rows, but here we need
        predictions for ALL rows. transform() only fills NaN, so we
        temporarily NaN out the target on a copy, run transform, and
        read back the filled values.
        """
        df_query = df.copy()
        df_query[self.target_col] = pd.NA
        df_query[self.target_col] = pd.to_numeric(
            df_query[self.target_col], errors="coerce",
        )
        
        filled = kriging_imputer.transform(df_query)
        result = df.copy()
        result[KRIGING_PRIOR_COL] = filled[self.target_col].astype(float).to_numpy()

        return result

    def fit(
        self, df: pd.DataFrame, lat: str, lon: str, target: str, **kwargs,
    ) -> "KrigingPriorRegression":
        self.lat_col, self.lon_col, self.target_col = lat, lon, target

        df_observed = df.dropna(subset=[target]).reset_index(drop=True)

        self.kriging_ = KrigingImputer(**self.kriging_kwargs)
        self.kriging_.fit(df, lat=lat, lon=lon, target=target)

        df_with_prior = df_observed.copy()
        if self.oof_folds is not None and len(df_observed) >= self.oof_folds:
            df_with_prior[KRIGING_PRIOR_COL] = self._compute_oof_prior(
                df_observed, lat, lon, target, n_folds=self.oof_folds,
            )
        else:
            df_with_prior = self._add_kriging_prior(df_observed, self.kriging_)

        gbm_features = list(self.feature_cols) + [KRIGING_PRIOR_COL]
        self.gbm_ = GBMImputer(feature_cols=gbm_features, **self.gbm_kwargs)
        self.gbm_.fit(df_with_prior, lat=lat, lon=lon, target=target)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()

        kriging_filled = self.kriging_.transform(df)

        df_with_prior = df.copy()
        df_with_prior[KRIGING_PRIOR_COL] = kriging_filled[self.target_col].astype(float).to_numpy()

        result = self.gbm_.transform(df_with_prior)

        if KRIGING_PRIOR_COL in result.columns:
            result = result.drop(columns=[KRIGING_PRIOR_COL])

        return result
