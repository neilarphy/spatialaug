"""Time and memory profilers for imputers.

The benchmark already records fit_time_sec and transform_time_sec.
This module adds peak memory measurement via the standard-library
tracemalloc (no external dependencies) and exposes:

- MethodProfiler — a context manager for measuring a single arbitrary
  block of code.
- profile_fit_transform — one-shot wrapper that runs fit then
  transform on a given imputer and returns both timings and peak
  memory for each phase.
- profile_scaling — runs an imputer on increasing subsample sizes to
  estimate empirical complexity.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd

from spatialaug.imputers.base import Imputer


@dataclass
class ProfileResult:
    """Result of a single fit + transform profiling cycle."""
    fit_time_sec: float
    transform_time_sec: float
    fit_memory_peak_mb: float
    transform_memory_peak_mb: float
    n_train: int
    n_predict: int
    method_name: str

    def to_dict(self) -> dict:
        return {
            "method": self.method_name,
            "n_train": self.n_train,
            "n_predict": self.n_predict,
            "fit_time_sec": self.fit_time_sec,
            "transform_time_sec": self.transform_time_sec,
            "fit_memory_peak_mb": self.fit_memory_peak_mb,
            "transform_memory_peak_mb": self.transform_memory_peak_mb,
            "throughput_predict_per_sec": (
                self.n_predict / self.transform_time_sec
                if self.transform_time_sec > 0 else float("nan")
            ),
        }


class MethodProfiler:
    """Context manager for measuring wall-clock time and peak heap usage.

    Wall-clock time is measured with time.perf_counter; peak memory
    with the standard-library tracemalloc.

    If tracemalloc was NOT tracing on entry, the profiler turns it on
    and stops it again on exit. If tracemalloc was ALREADY tracing
    (started by another profiler, a debugger, an outer block of the
    same kind, ...), the profiler simply resets the peak counter on
    entry and leaves the tracing running on exit — so it never
    interferes with surrounding instrumentation.

    Examples
    --------
    >>> with MethodProfiler() as p:
    ...     model.fit(X, y)
    >>> print(f"fit took {p.elapsed_sec:.2f}s, peak {p.peak_mb:.1f} MB")
    """

    def __init__(self, label: str = "anonymous") -> None:
        self.label = label
        self.elapsed_sec: float = 0.0
        self.peak_mb: float = 0.0
        self._start_time: float | None = None
        self._owns_tracing: bool = False

    def __enter__(self) -> "MethodProfiler":
        self._owns_tracing = not tracemalloc.is_tracing()

        if self._owns_tracing:
            tracemalloc.start()
        else:
            tracemalloc.reset_peak()
        
        self._start_time = time.perf_counter()
        
        return self

    def __exit__(self, *_exc_info: object) -> None:
        del _exc_info 
        
        if self._start_time is not None:
            self.elapsed_sec = time.perf_counter() - self._start_time
        _, peak = tracemalloc.get_traced_memory()
        
        self.peak_mb = peak / 1e6
        
        if self._owns_tracing:
            tracemalloc.stop()
            self._owns_tracing = False


def profile_fit_transform(
    imputer: Imputer,
    df_train: pd.DataFrame,
    df_predict: pd.DataFrame,
    *,
    lat: str,
    lon: str,
    target: str,
    method_name: str | None = None,
) -> ProfileResult:
    """Profile a single imputer's fit and transform pass on the given data.

    Returns timings and peak memory for each phase wrapped in a
    ProfileResult.

    Parameters
    ----------
    imputer : Imputer
        Already-instantiated imputer (e.g. MeanImputer(strategy="median")).
    df_train : pd.DataFrame
        Data used to fit the imputer. Rows with NaN in target are
        dropped inside fit().
    df_predict : pd.DataFrame
        Data fed to transform. Rows with NaN in target are the ones
        the imputer fills.
    lat, lon, target : str
        Column names passed to the imputer.
    method_name : str or None, default=None
        Label used in the result. Defaults to the imputer's class name.
    """
    name = method_name or type(imputer).__name__

    with MethodProfiler("fit") as fit_prof:
        imputer.fit(df_train, lat=lat, lon=lon, target=target)

    with MethodProfiler("transform") as trans_prof:
        _ = imputer.transform(df_predict)

    n_predict = int(df_predict[target].isna().sum())
    return ProfileResult(
        method_name=name,
        n_train=int(len(df_train.dropna(subset=[target]))),
        n_predict=n_predict,
        fit_time_sec=fit_prof.elapsed_sec,
        transform_time_sec=trans_prof.elapsed_sec,
        fit_memory_peak_mb=fit_prof.peak_mb,
        transform_memory_peak_mb=trans_prof.peak_mb,
    )


def profile_scaling(
    imputer_factory: Callable[[], Imputer],
    df: pd.DataFrame,
    *,
    lat: str, lon: str, target: str,
    sizes: list[int] | None = None,
    mask_ratio: float = 0.30,
    random_state: int = 42,
    method_name: str | None = None,
) -> pd.DataFrame:
    """Empirical scaling profile across increasing dataset sizes.

    For each size N, samples N rows from df, masks mask_ratio of the
    target column, then runs fit + transform via profile_fit_transform.
    Useful for distinguishing empirical complexity classes (O(n log n)
    vs O(n^2)) by inspecting how fit_time and memory grow with N.

    Parameters
    ----------
    imputer_factory : callable
        Zero-arg callable producing a fresh Imputer instance per size,
        so each measurement starts from a clean state.
    df : pd.DataFrame
        Source dataset to subsample from.
    lat, lon, target : str
        Column names passed to the imputer.
    sizes : list of int, optional
        Sample sizes to test. Defaults to [100, 250, 500, 1000, 2000].
        Sizes larger than len(df) are skipped.
    mask_ratio : float, default=0.30
        Fraction of target values masked in each subsample.
    random_state : int, default=42
        Seed for subsampling and masking.
    method_name : str or None, default=None
        Label forwarded to profile_fit_transform.

    Returns
    -------
    pd.DataFrame
        One row per size with columns: n, method, n_train, n_predict,
        fit_time_sec, transform_time_sec, fit_memory_peak_mb,
        transform_memory_peak_mb, throughput_predict_per_sec, error.
    """
    
    if sizes is None:
        sizes = [100, 250, 500, 1000, 2000]
    
    rng = np.random.default_rng(random_state)
    rows = []
    
    for n in sizes:
        if n > len(df):
            continue
        sub = df.sample(n=n, random_state=random_state).reset_index(drop=True)
        mask_idx = rng.choice(len(sub), int(len(sub) * mask_ratio), replace=False)
        sub_masked = sub.copy()
        sub_masked.loc[mask_idx, target] = np.nan

        imp = imputer_factory()
        try:
            res = profile_fit_transform(
                imp, sub_masked, sub_masked,
                lat=lat, lon=lon, target=target,
                method_name=method_name,
            )
            rows.append({"n": n, **res.to_dict(), "error": None})
        except Exception as exc:
            rows.append({"n": n, "error": f"{type(exc).__name__}: {exc}"})
    
    return pd.DataFrame(rows)
