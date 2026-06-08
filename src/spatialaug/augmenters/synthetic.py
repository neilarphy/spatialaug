"""Augmenter — synthetic geo-point generation for data augmentation.

This module implements the classical ML notion of data augmentation
(albumentations / nlpaug / SMOTE / CTGAN family): generate new samples
at new locations and predict their target via a built-in Imputer.
The synthetic rows are concatenated with the observed training set,
producing an augmented dataset of size len(observed) + len(synthetic).

Because the ground truth at synthetic locations is unknown, the quality
of the augmentation can only be measured downstream (e.g. ΔF1 on a
classifier trained on the augmented set).

Four strategies are supported:

1. regular_grid — uniform lat/lon grid with step grid_step_km.
2. density_fill — synthetic points in sparse regions (candidates
   farthest from any observed point are kept).
3. mixup — convex combination of random pairs of observed points
   (coordinates and features mixed with the same α ~ Beta(0.5, 0.5)).
4. jitter — observed coordinates perturbed by Gaussian noise with
   standard deviation jitter_km (converted to degrees via the local
   latitude).

To add a new strategy: implement a method with the same signature as
the existing _generate_* methods (returning (coords, feats_or_None)),
add its name to STRATEGY_NAMES, and register it in the _strategies()
dispatch table.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from spatialaug.imputers.base import Imputer
from spatialaug.utils.geo import km_to_deg

STRATEGY_NAMES = ("regular_grid", "density_fill", "mixup", "jitter")
Strategy = Literal["regular_grid", "density_fill", "mixup", "jitter"]


class Augmenter:
    """Generate synthetic geo-points and predict their target via an Imputer.

    Parameters
    ----------
    imputer : Imputer
        Base imputer used to predict the target at synthetic locations
        (e.g. KrigingImputer, UniversalKrigingImputer for KED,
        GBMImputer, TabPFNImputer).
    strategy : {"regular_grid", "density_fill", "mixup", "jitter"}
        How the synthetic coordinates are generated. See module docstring.
    n_synthetic : int or float, default=0.5
        If int — exact number of synthetic points to produce.
        If float in (0, 1) — fraction of the observed set
        (e.g. 0.5 means 50 % synthetic relative to observed).
    grid_step_km : float, default=1.0
        Grid step for the regular_grid strategy. Not used by other
        strategies.
    jitter_km : float, default=0.5
        Standard deviation of the Gaussian noise added to coordinates
        in the jitter strategy.
    min_distance_km : float, default=0.5
        Minimum allowed distance between a synthetic point and any row
        in exclude_df (typically the test set). Synthetic points
        closer than this threshold are dropped in fit_transform to
        prevent train/test leakage.
    random_state : int, default=42
        Seed for all random sampling (grid subsampling, mixup pairs,
        jitter noise).

    Examples
    --------
    >>> from spatialaug import KrigingImputer
    >>> from spatialaug.augmenters import Augmenter
    >>> aug = Augmenter(KrigingImputer(), strategy="regular_grid",
    ...                 n_synthetic=0.5)
    >>> df_aug = aug.fit_transform(
    ...     df_train, lat="centroid_lat", lon="centroid_lon",
    ...     target="avg_bill", feature_cols=["kkt_count", "is_mall"],
    ... )
    >>> # df_aug = df_train + synthetic rows (with a _is_synthetic flag)
    """

    def __init__(
        self,
        imputer: Imputer,
        strategy: Strategy = "regular_grid",
        n_synthetic: int | float = 0.5,
        grid_step_km: float = 1.0,
        jitter_km: float = 0.5,
        min_distance_km: float = 0.5,
        random_state: int = 42,
    ) -> None:

        if strategy not in STRATEGY_NAMES:
            raise ValueError(
                f"unknown strategy {strategy!r}, expected one of {STRATEGY_NAMES}"
            )

        self.imputer = imputer
        self.strategy = strategy
        self.n_synthetic = n_synthetic
        self.grid_step_km = float(grid_step_km)
        self.jitter_km = float(jitter_km)
        self.min_distance_km = float(min_distance_km)
        self.random_state = int(random_state)

    def _strategies(self) -> dict:
        """Map strategy name → bound generator method.

        All generators share the signature
        (df, lat, lon, feature_cols, n_target) -> (coords, feats_or_None).
        Strategies that return None for feats fall back to
        _assign_features (nearest-observed copy).
        """
        return {
            "regular_grid": self._generate_regular_grid,
            "density_fill": self._generate_density_fill,
            "mixup":        self._generate_mixup,
            "jitter":       self._generate_jitter,
        }

    def _n_target(self, n_observed: int) -> int:
        """Resolve n_synthetic (int or fraction) into an absolute count."""

        if isinstance(self.n_synthetic, float) and 0 < self.n_synthetic < 1:
            return int(n_observed * self.n_synthetic)

        return int(self.n_synthetic)

    def _generate_regular_grid(
        self, df: pd.DataFrame, lat: str, lon: str,
        feature_cols: list[str], n_target: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Build a uniform lat/lon grid and randomly subsample to n_target.

        Returns coords only; features are assigned later via
        _assign_features (nearest observed).
        """
        del feature_cols  # unused — features assigned via _assign_features
        center_lat = df[lat].mean()
        deg_lat, deg_lon = km_to_deg(center_lat, self.grid_step_km)
        lat_min, lat_max = df[lat].min(), df[lat].max()
        lon_min, lon_max = df[lon].min(), df[lon].max()

        lats = np.arange(lat_min, lat_max + deg_lat, deg_lat)
        lons = np.arange(lon_min, lon_max + deg_lon, deg_lon)
        grid_lat, grid_lon = np.meshgrid(lats, lons)
        coords = np.column_stack([grid_lat.ravel(), grid_lon.ravel()])
        rng = np.random.default_rng(self.random_state)

        if len(coords) > n_target:
            idx = rng.choice(len(coords), n_target, replace=False)
            coords = coords[idx]

        return coords, None

    def _generate_density_fill(
        self, df: pd.DataFrame, lat: str, lon: str,
        feature_cols: list[str], n_target: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Place synthetic points where the observed set is sparsest.

        Draw a dense pool of uniform candidates inside the bounding
        box, then keep the n_target candidates that are farthest from
        any observed point (largest 1-NN distance). Features assigned
        later via _assign_features.
        """
        del feature_cols  # unused — features assigned via _assign_features
        lat_min, lat_max = df[lat].min(), df[lat].max()
        lon_min, lon_max = df[lon].min(), df[lon].max()

        n_candidates = max(n_target * 10, 1000)
        rng = np.random.default_rng(self.random_state)
        cand_lat = rng.uniform(lat_min, lat_max, n_candidates)
        cand_lon = rng.uniform(lon_min, lon_max, n_candidates)
        cand = np.column_stack([cand_lat, cand_lon])

        obs = df[[lat, lon]].to_numpy()
        tree = KDTree(obs)
        dist, _ = tree.query(cand, k=1)

        idx = np.argsort(-dist)[:n_target]

        return cand[idx], None

    def _generate_mixup(
        self, df: pd.DataFrame, lat: str, lon: str,
        feature_cols: list[str], n_target: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Mixup: convex combination of random pairs of observed points.

        Coordinates and features are mixed with the same per-row
        α ~ Beta(0.5, 0.5), which favours extremes (α close to 0 or 1)
        and therefore stays close to the observed manifold while still
        producing new locations. Returns the mixed feature matrix
        directly (no _assign_features fallback).
        """
        rng = np.random.default_rng(self.random_state)
        n_obs = len(df)

        if n_obs < 2:
            return np.empty((0, 2)), np.empty((0, len(feature_cols)))

        idx1 = rng.integers(0, n_obs, n_target)
        idx2 = rng.integers(0, n_obs, n_target)

        alpha = rng.beta(0.5, 0.5, n_target).reshape(-1, 1)
        coords_1 = df[[lat, lon]].iloc[idx1].to_numpy()
        coords_2 = df[[lat, lon]].iloc[idx2].to_numpy()
        synthetic_coords = alpha * coords_1 + (1 - alpha) * coords_2

        if feature_cols:
            feats_1 = df[feature_cols].iloc[idx1].to_numpy(dtype=float)
            feats_2 = df[feature_cols].iloc[idx2].to_numpy(dtype=float)
            synthetic_feats = alpha * feats_1 + (1 - alpha) * feats_2
        else:
            synthetic_feats = np.empty((n_target, 0))

        return synthetic_coords, synthetic_feats

    def _generate_jitter(
        self, df: pd.DataFrame, lat: str, lon: str,
        feature_cols: list[str], n_target: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Resample observed coordinates and add Gaussian noise.

        Features assigned later via _assign_features.
        """
        del feature_cols  # unused — features assigned via _assign_features
        center_lat = df[lat].mean()
        deg_lat, deg_lon = km_to_deg(center_lat, self.jitter_km)

        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(df), n_target, replace=True)

        coords = df[[lat, lon]].iloc[idx].to_numpy().copy()
        coords[:, 0] += rng.normal(0, deg_lat, n_target)
        coords[:, 1] += rng.normal(0, deg_lon, n_target)

        return coords, None

    def _assign_features(
        self, synthetic_coords: np.ndarray,
        observed_df: pd.DataFrame, lat: str, lon: str,
        feature_cols: list[str],
    ) -> np.ndarray:
        """Copy features from the nearest observed point to each synthetic point."""

        if not feature_cols:
            return np.empty((len(synthetic_coords), 0))

        obs_coords = observed_df[[lat, lon]].to_numpy()

        tree = KDTree(obs_coords)
        _, idx = tree.query(synthetic_coords, k=1)

        return observed_df[feature_cols].iloc[idx].to_numpy(dtype=float)

    def _filter_too_close(
        self, synthetic_coords: np.ndarray,
        exclude_df: pd.DataFrame | None,
        lat: str, lon: str,
    ) -> np.ndarray:
        """Boolean mask: True for synthetic points far enough from exclude_df.

        Used to prevent leakage: synthetic points closer than
        min_distance_km to any row in exclude_df (typically the test
        set) are dropped.
        """

        if exclude_df is None or len(exclude_df) == 0:
            return np.ones(len(synthetic_coords), dtype=bool)

        center_lat = synthetic_coords[:, 0].mean()
        deg_lat, deg_lon = km_to_deg(center_lat, self.min_distance_km)

        min_deg = math.sqrt(deg_lat ** 2 + deg_lon ** 2) / math.sqrt(2)
        excl_coords = exclude_df[[lat, lon]].to_numpy()

        tree = KDTree(excl_coords)
        dist, _ = tree.query(synthetic_coords, k=1)

        return dist > min_deg

    def fit_transform(
        self,
        df_train: pd.DataFrame,
        *,
        lat: str,
        lon: str,
        target: str,
        feature_cols: list[str] | None = None,
        exclude_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate synthetic points and predict their target via the Imputer.

        Pipeline:

        1. Drop rows with missing target from df_train, call it observed.
        2. Generate synthetic coordinates (and possibly features) via
           the strategy registered under self.strategy.
        3. Drop synthetic points too close to exclude_df (leakage guard).
        4. If the strategy didn't produce features, assign them from
           the nearest observed point.
        5. Fit the imputer on observed and predict the target at the
           synthetic locations.
        6. Concatenate observed with the imputed synthetic rows and
           add a boolean _is_synthetic column.

        Parameters
        ----------
        df_train : pd.DataFrame
            Training set. Rows with missing target are excluded from
            fitting and from the returned observed block.
        lat, lon : str
            Column names of the geographic coordinates.
        target : str
            Column to be predicted at synthetic locations.
        feature_cols : list[str] or None, default=None
            Additional feature columns to copy/interpolate onto the
            synthetic rows. Defaults to an empty list.
        exclude_df : pd.DataFrame or None, default=None
            Rows that synthetic points must not be too close to
            (typically the held-out test set). See min_distance_km.

        Returns
        -------
        pd.DataFrame
            Observed rows followed by synthetic rows. Synthetic rows
            have the imputed target and copied/interpolated features.
            A boolean _is_synthetic column marks which rows are new.
            If too few observed points are available (< 10) or all
            synthetic points are filtered out, observed is returned
            unchanged.
        """

        if feature_cols is None:
            feature_cols = []
        observed = df_train.dropna(subset=[target]).reset_index(drop=True)

        if len(observed) < 10:
            return observed

        n_target = self._n_target(len(observed))

        generator = self._strategies()[self.strategy]
        synthetic_coords, synthetic_feats = generator(
            observed, lat, lon, feature_cols, n_target,
        )

        keep = self._filter_too_close(synthetic_coords, exclude_df, lat, lon)
        synthetic_coords = synthetic_coords[keep]
        if synthetic_feats is not None:
            synthetic_feats = synthetic_feats[keep]

        if len(synthetic_coords) == 0:
            return observed

        if synthetic_feats is None:
            synthetic_feats = self._assign_features(
                synthetic_coords, observed, lat, lon, feature_cols)

        self.imputer.fit(observed, lat=lat, lon=lon, target=target)
        synth_df = pd.DataFrame({
            lat: synthetic_coords[:, 0],
            lon: synthetic_coords[:, 1],
            target: np.nan,
        })

        for i, col in enumerate(feature_cols):
            synth_df[col] = synthetic_feats[:, i] if i < synthetic_feats.shape[1] else np.nan
        synth_filled = self.imputer.transform(synth_df)

        cols = [lat, lon, target] + list(feature_cols)
        combined = pd.concat([observed[cols], synth_filled[cols]], ignore_index=True)
        combined["_is_synthetic"] = (
            [False] * len(observed) + [True] * len(synth_filled)
        )

        return combined
