"""Benchmark runner — orchestrates fit+transform of many imputers across mechanisms.

Three entry points:
- run_benchmark: every imputer x every mechanism on a synthetic mask of df.
- run_benchmark_multi_seed: wraps run_benchmark across N seeds.
- run_benchmark_spatial_cv: spatial-block CV evaluation where test rows are
  geographically grouped, so kriging is forced to extrapolate beyond the
  training cluster — a noticeably harder and more honest setup.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from spatialaug.benchmark.missingness import MissingnessSimulator
from spatialaug.imputers.base import Imputer
from spatialaug.metrics.cv import SpatialBlockCV


def _fit_score_imputer(
    factory: Callable[[], Imputer],
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    eval_mask: np.ndarray,
    true_values: np.ndarray,
    *,
    target: str,
    lat: str,
    lon: str,
) -> dict:
    """Instantiate one imputer, fit on train_df, transform eval_df, score on the mask.

    Returns a dict with the canonical benchmark fields used by every
    public runner: mae, rmse, fit_time_sec, transform_time_sec, error.
    Exceptions during fit / transform are caught and reported as NaN
    metrics with a populated error string, so a single failing
    method does not abort the whole benchmark.
    """

    out: dict = {
        "mae": np.nan,
        "rmse": np.nan,
        "fit_time_sec": np.nan,
        "transform_time_sec": np.nan,
        "error": None,
    }
    
    try:
        imp = factory()
        t0 = time.perf_counter()
        imp.fit(train_df, lat=lat, lon=lon, target=target)
        out["fit_time_sec"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        filled = imp.transform(eval_df)
        out["transform_time_sec"] = time.perf_counter() - t0

        predicted = filled.loc[eval_mask, target].to_numpy(dtype=float)
        errors = predicted - true_values
        out["mae"] = float(np.mean(np.abs(errors)))
        out["rmse"] = float(np.sqrt(np.mean(errors ** 2)))
    
    except Exception as exc:  
        out["error"] = f"{type(exc).__name__}: {exc}"
    
    return out


def run_benchmark(
    df: pd.DataFrame,
    imputer_factories: dict[str, Callable[[], Imputer]],
    target: str,
    lat: str,
    lon: str,
    mechanisms: Sequence[str] = ("mcar", "diffuse_mnar", "focused_mnar"),
    missingness_ratio: float = 0.3,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run each imputer against each missingness mechanism on df.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (no artificial NaN yet).
    imputer_factories : dict[str, Callable[[], Imputer]]
        Map of {method_name: zero-arg factory}. A factory is required
        (not an instance) because each run needs a fresh, unfitted
        imputer.
    target : str
        Target column to mask and reconstruct.
    lat, lon : str
        Coordinate column names.
    mechanisms : sequence of str
        Mechanisms to evaluate.
    missingness_ratio : float, default=0.3
        Fraction of observable rows to mask.
    random_state : int, default=42
        Seed passed to MissingnessSimulator.
    verbose : bool, default=True
        Print progress while running.

    Returns
    -------
    pd.DataFrame
        Columns: imputer, mechanism, n_missing, missingness_ratio,
        mae, rmse, fit_time_sec, transform_time_sec, error.

    Examples
    --------
    >>> from spatialaug import MeanImputer, IDWImputer
    >>> results = run_benchmark(
    ...     df, target="price", lat="geo_lat", lon="geo_lon",
    ...     imputer_factories={
    ...         "mean": lambda: MeanImputer(strategy="median"),
    ...         "idw": lambda: IDWImputer(power=2, n_neighbors=10),
    ...     },
    ... )
    """
    
    results = []

    for mech in mechanisms:
        if verbose:
            print(f"\n[benchmark] mechanism = {mech}, ratio = {missingness_ratio}")

        sim = MissingnessSimulator(
            mechanism=mech,
            ratio=missingness_ratio,
            random_state=random_state,
        )
        df_masked, mask = sim.apply(df, target_col=target, lat_col=lat, lon_col=lon)
        true_values = df.loc[mask, target].to_numpy(dtype=float)
        n_missing = int(mask.sum())

        for name, factory in imputer_factories.items():
            if verbose:
                print(f"  [{name}] ", end="", flush=True)
            row = {
                "imputer": name,
                "mechanism": mech,
                "n_missing": n_missing,
                "missingness_ratio": missingness_ratio,
            }
            scores = _fit_score_imputer(
                factory, df_masked, df_masked, mask, true_values,
                target=target, lat=lat, lon=lon,
            )
            row.update(scores)
            if verbose:
                if row["error"] is None:
                    print(
                        f"MAE={row['mae']:.4g}, RMSE={row['rmse']:.4g}, "
                        f"fit={row['fit_time_sec']:.2f}s, "
                        f"transform={row['transform_time_sec']:.2f}s"
                    )
                else:
                    print(f"FAILED ({row['error']})")
            results.append(row)

    return pd.DataFrame(results)


def run_benchmark_multi_seed(
    df: pd.DataFrame,
    imputer_factories: dict[str, Callable[[], Imputer]],
    target: str,
    lat: str,
    lon: str,
    mechanisms: Sequence[str] = ("mcar", "diffuse_mnar", "focused_mnar"),
    missingness_ratio: float = 0.3,
    seeds: Sequence[int] = (42, 1, 2, 3, 7),
    verbose: bool = True,
) -> pd.DataFrame:
    """Run ``run_benchmark`` for each seed and concatenate the results.

    Each row carries the ``seed`` it came from so downstream analysis
    can compute mean +/- std across seeds and tell real findings from
    artefacts of a single random hold-out.

    Why it matters: on small regions (e.g. Yakutsk ~60 hex, ~18 test
    points) a single-seed MAE can swing +/- 15-20%. Only differences
    that survive 3-5 seeds are scientifically meaningful.

    Parameters
    ----------
    seeds : sequence of int, default=(42, 1, 2, 3, 7)
        Random states used for the repeated runs.

    Other parameters are forwarded to ``run_benchmark``.

    Returns
    -------
    pd.DataFrame
        Same columns as ``run_benchmark`` plus a ``seed`` column.
    """
    
    all_results: list[pd.DataFrame] = []
    
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'#' * 60}\n# SEED = {seed} ({i + 1}/{len(seeds)})\n{'#' * 60}")
        results = run_benchmark(
            df=df,
            target=target,
            lat=lat,
            lon=lon,
            imputer_factories=imputer_factories,
            mechanisms=mechanisms,
            missingness_ratio=missingness_ratio,
            random_state=seed,
            verbose=verbose,
        )
        results["seed"] = seed
        all_results.append(results)
    
    return pd.concat(all_results, ignore_index=True)


def run_benchmark_spatial_cv(
    df: pd.DataFrame,
    imputer_factories: dict[str, Callable[[], Imputer]],
    target: str,
    lat: str,
    lon: str,
    n_splits: int = 5,
    block_size_km: float | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Benchmark under spatial block cross-validation.

    Unlike ``run_benchmark`` (random hold-out where kriging almost
    always has training neighbours close to each test point), this
    runner groups test rows into geographic blocks so kriging is
    forced to extrapolate beyond the training cluster — a much more
    honest evaluation in the presence of spatial autocorrelation
    (Roberts et al. 2017; Meyer et al. 2019).

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    block_size_km : float, optional
        Block size in km. None triggers the auto heuristic
        bbox_diagonal / (n_splits * 2).
    random_state : int, default=42
        Seed for block shuffling.

    Other parameters are forwarded as in ``run_benchmark``.

    Returns
    -------
    pd.DataFrame
        Columns: imputer, fold, mechanism="spatial_cv", n_test, mae,
        rmse, fit_time_sec, transform_time_sec, error.
    """
    cv = SpatialBlockCV(
        n_splits=n_splits, block_size_km=block_size_km, random_state=random_state,
    )
    results: list[dict] = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(
        cv.split(df, lat_col=lat, lon_col=lon)
    ):
        if verbose:
            print(
                f"\n[spatial_cv] fold {fold_idx + 1}/{n_splits}: "
                f"train={len(train_idx)}, test={len(test_idx)}"
            )
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        true_values = test_df[target].to_numpy(dtype=float)

        test_df_masked = test_df.copy()
        test_df_masked[target] = np.nan
        combined = pd.concat([train_df, test_df_masked], ignore_index=True)
        test_mask_in_combined = np.zeros(len(combined), dtype=bool)
        test_mask_in_combined[len(train_df):] = True

        for name, factory in imputer_factories.items():
            if verbose:
                print(f"  [{name}] ", end="", flush=True)
            row = {
                "imputer": name,
                "fold": fold_idx,
                "mechanism": "spatial_cv",
                "n_test": int(len(test_idx)),
            }
            scores = _fit_score_imputer(
                factory, combined, combined, test_mask_in_combined, true_values,
                target=target, lat=lat, lon=lon,
            )
            row.update(scores)
            if verbose:
                if row["error"] is None:
                    print(f"MAE={row['mae']:.4g}, fit={row['fit_time_sec']:.2f}s")
                else:
                    print(f"FAILED ({row['error']})")
            results.append(row)

    return pd.DataFrame(results)
