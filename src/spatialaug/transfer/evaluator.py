"""TransferEvaluator — cross-region transferability evaluation.

One call:
1. Fit the imputer on the source region.
2. Apply it to the target region, where ground truth is known via
   synthetic masking.
3. Compare against the local baseline (full_refit: fit a fresh
   imputer on the target's own observed rows).
4. Report ``transfer_stability = MAE(zero_shot) / MAE(full_refit)``.

Multi-seed evaluation is built in so that each (source, target,
mechanism) cell ships with mean and std across the seeds.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from spatialaug.benchmark.missingness import MissingnessSimulator
from spatialaug.imputers.base import Imputer


def _error_dict(exc: Exception) -> dict:
    """Standard NaN-result dict tagged with a short error description."""
    return {
        "mae": np.nan,
        "rmse": np.nan,
        "fit_time": np.nan,
        "transform_time": np.nan,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _finite_mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    """Mean and std of a per-seed numeric field, ignoring NaN/None.

    Returns (NaN, 0.0) if no finite values are present, and (mean, 0.0)
    when only a single finite value remains (std is undefined).
    """
    
    values = np.array(
        [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])],
        dtype=float,
    )
    
    if len(values) == 0:
        return float("nan"), 0.0
    
    mean = float(values.mean())
    std = float(values.std()) if len(values) > 1 else 0.0
    
    return mean, std


@dataclass
class TransferResult:
    """Aggregated result of one (source, target, mechanism) cell.

    The aggregated fields (mean, std) are computed across the seeds in
    ``per_seed_rows``. ``per_seed_rows`` itself keeps the raw per-seed
    measurements for downstream inspection.
    """
    source_name: str
    target_name: str
    target_col: str
    method_name: str
    mechanism: str
    n_source: int
    n_target: int
    n_masked_target: int

    mae_zero_shot_mean: float
    mae_zero_shot_std: float
    mae_full_refit_mean: float
    mae_full_refit_std: float
    transfer_stability_mean: float
    transfer_stability_std: float

    n_seeds: int
    per_seed_rows: list[dict] = field(default_factory=list)

    def to_row(self) -> dict:
        """Flatten to a single dict suitable for pd.DataFrame aggregation."""

        return {
            "source": self.source_name,
            "target": self.target_name,
            "fns_target": self.target_col,
            "method": self.method_name,
            "mechanism": self.mechanism,
            "n_source": self.n_source,
            "n_target": self.n_target,
            "n_masked": self.n_masked_target,
            "mae_zero_shot": self.mae_zero_shot_mean,
            "mae_zero_shot_std": self.mae_zero_shot_std,
            "mae_full_refit": self.mae_full_refit_mean,
            "mae_full_refit_std": self.mae_full_refit_std,
            "transfer_stability": self.transfer_stability_mean,
            "transfer_stability_std": self.transfer_stability_std,
            "n_seeds": self.n_seeds,
        }


class TransferEvaluator:
    """Multi-seed transferability evaluation for a single imputer.

    Parameters
    ----------
    imputer_factory : callable
        Zero-arg callable producing a fresh Imputer instance per fit
        (e.g. ``lambda: UniversalKrigingImputer(feature_cols=feats)``).
        A factory rather than an instance ensures every seed/strategy
        starts from a clean state.
    method_name : str, default="imputer"
        Display name used in reports and DataFrames.
    lat_col, lon_col : str
        Coordinate column names passed through to the imputer.
    target_col : str
        Target column to be masked and reconstructed.
    seeds : list or tuple of int, default=(1, 2, 3, 7, 42)
        Random seeds for the MissingnessSimulator. The mean/std of all
        metrics are reported across these seeds.
    mechanisms : list or tuple of str, default=("mcar", "diffuse_mnar", "focused_mnar")
        Missingness mechanisms to stress-test the imputer with.
    missingness_ratio : float, default=0.30
        Fraction of target rows masked in the target region.

    Examples
    --------
    >>> from spatialaug import UniversalKrigingImputer
    >>> from spatialaug.transfer import TransferEvaluator
    >>> evaluator = TransferEvaluator(
    ...     imputer_factory=lambda: UniversalKrigingImputer(feature_cols=["kkt_count"]),
    ...     method_name="ked",
    ...     lat_col="centroid_lat", lon_col="centroid_lon",
    ...     target_col="avg_bill",
    ...     seeds=[1, 2, 3],
    ... )
    >>> results = evaluator.evaluate(
    ...     source_df=moscow, target_df=yakutsk,
    ...     source_name="moscow", target_name="yakutsk",
    ... )
    >>> for r in results:
    ...     print(f"{r.mechanism}: stability = {r.transfer_stability_mean:.2f}")
    """

    def __init__(
        self,
        imputer_factory: Callable[[], Imputer],
        method_name: str = "imputer",
        *,
        lat_col: str = "centroid_lat",
        lon_col: str = "centroid_lon",
        target_col: str = "avg_bill",
        seeds: list[int] | tuple[int, ...] = (1, 2, 3, 7, 42),
        mechanisms: list[str] | tuple[str, ...] = (
            "mcar", "diffuse_mnar", "focused_mnar",
        ),
        missingness_ratio: float = 0.30,
    ) -> None:
        
        self.imputer_factory = imputer_factory
        self.method_name = method_name
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.target_col = target_col
        self.seeds = list(seeds)
        self.mechanisms = list(mechanisms)
        self.missingness_ratio = float(missingness_ratio)

    def _fit_and_score(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        eval_mask: np.ndarray,
        true_values: np.ndarray,
    ) -> dict:
        """Fit the imputer on train_df, predict eval_df, score on the masked rows.

        Returns a dict with keys: mae, rmse, fit_time, transform_time.
        """
        
        imp = self.imputer_factory()
        t0 = time.perf_counter()
        imp.fit(train_df, lat=self.lat_col, lon=self.lon_col,
                target=self.target_col)
        fit_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        
        filled = imp.transform(eval_df)
        transform_time = time.perf_counter() - t0
        pred = filled[self.target_col].to_numpy(float)[eval_mask]
        valid = np.isfinite(pred) & np.isfinite(true_values)
        
        if valid.sum() == 0:
            return {
                "mae": np.nan, "rmse": np.nan,
                "fit_time": fit_time, "transform_time": transform_time,
            }
        
        err = pred[valid] - true_values[valid]
        
        return {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "fit_time": fit_time,
            "transform_time": transform_time,
        }

    def evaluate(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        *,
        source_name: str = "source",
        target_name: str = "target",
    ) -> list[TransferResult]:
        """Multi-seed x multi-mechanism transfer evaluation for one pair.

        For every (mechanism, seed) the target is masked, the imputer
        is fit on the source (zero-shot) and refit on the target's
        observed rows (full_refit), and MAE is computed against the
        held-out true values. Results are aggregated per mechanism.

        Returns
        -------
        list[TransferResult]
            One element per mechanism, with mean/std aggregated across
            seeds.
        """
        
        results = []
        
        for mech in self.mechanisms:
            per_seed_rows = []
            target_mask = np.zeros(len(target_df), dtype=bool)
            for seed in self.seeds:
                sim = MissingnessSimulator(
                    mechanism=mech, ratio=self.missingness_ratio,
                    random_state=seed,
                )
                target_masked, target_mask = sim.apply(
                    target_df, target_col=self.target_col,
                    lat_col=self.lat_col, lon_col=self.lon_col,
                )
                true_values = (
                    target_df[self.target_col].to_numpy(float)[target_mask]
                )

                try:
                    z = self._fit_and_score(
                        source_df, target_masked, target_mask, true_values,
                    )
                except Exception as exc:
                    z = _error_dict(exc)

                try:
                    target_observed = target_masked.dropna(
                        subset=[self.target_col],
                    )
                    f = self._fit_and_score(
                        target_observed, target_masked, target_mask, true_values,
                    )
                except Exception as exc:
                    f = _error_dict(exc)

                f_mae = f.get("mae")
                if (
                    f_mae is not None
                    and np.isfinite(f_mae)
                    and f_mae > 1e-12
                ):
                    stability = z["mae"] / f_mae
                else:
                    stability = np.nan

                per_seed_rows.append({
                    "seed": seed,
                    "mae_zero_shot": z.get("mae"),
                    "mae_full_refit": f_mae,
                    "stability": stability,
                    "fit_time_zero_shot": z.get("fit_time"),
                    "fit_time_full_refit": f.get("fit_time"),
                })

            z_mean, z_std = _finite_mean_std(per_seed_rows, "mae_zero_shot")
            f_mean, f_std = _finite_mean_std(per_seed_rows, "mae_full_refit")
            stab_mean, stab_std = _finite_mean_std(per_seed_rows, "stability")

            results.append(TransferResult(
                source_name=source_name,
                target_name=target_name,
                target_col=self.target_col,
                method_name=self.method_name,
                mechanism=mech,
                n_source=int(len(source_df)),
                n_target=int(len(target_df)),
                n_masked_target=int(target_mask.sum()),
                mae_zero_shot_mean=z_mean,
                mae_zero_shot_std=z_std,
                mae_full_refit_mean=f_mean,
                mae_full_refit_std=f_std,
                transfer_stability_mean=stab_mean,
                transfer_stability_std=stab_std,
                n_seeds=len(per_seed_rows),
                per_seed_rows=per_seed_rows,
            ))
        return results

    def evaluate_matrix(
        self,
        cities: dict[str, pd.DataFrame],
        pairs: list[tuple[str, str]] | None = None,
    ) -> pd.DataFrame:
        """Full transferability matrix across (source -> target) pairs.

        Parameters
        ----------
        cities : dict[str, pd.DataFrame]
            Region name to DataFrame. Each DataFrame must contain the
            configured lat_col / lon_col / target_col columns.
        pairs : list of (source_name, target_name), optional
            Pairs to evaluate. When None, all ordered N x (N-1) pairs
            of distinct regions are evaluated. Pairs referencing
            missing region names are silently skipped.

        Returns
        -------
        pd.DataFrame
            One row per (pair, mechanism) cell with mean/std of MAE
            and transfer stability across the configured seeds.
        """
        
        if pairs is None:
            names = list(cities.keys())
            pairs = [(s, t) for s in names for t in names if s != t]

        
        rows = []
        
        for src, tgt in pairs:
            if src not in cities or tgt not in cities:
                continue
            results = self.evaluate(
                cities[src], cities[tgt],
                source_name=src, target_name=tgt,
            )
            for r in results:
                rows.append(r.to_row())
        
        return pd.DataFrame(rows)
