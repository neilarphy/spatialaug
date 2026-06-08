import time

import numpy as np
import pandas as pd

from spatialaug import IDWImputer, MeanImputer
from spatialaug.utils import MethodProfiler, profile_fit_transform
from spatialaug.utils.profiler import ProfileResult, profile_scaling


def test_method_profiler_measures_time():
    with MethodProfiler("test") as p:
        _ = sum(range(100000))
    assert p.elapsed_sec > 0.0001
    assert p.elapsed_sec < 5.0  


def test_method_profiler_measures_memory():
    with MethodProfiler("test") as p:
        _arr = np.zeros((1000, 1000), dtype=np.float64)
    assert p.peak_mb > 1.0


def test_method_profiler_context_releases():
    with MethodProfiler() as p:
        pass
    with MethodProfiler() as p2:
        pass
    assert p2.elapsed_sec >= 0


def test_profile_fit_transform_returns_result():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "lat": rng.uniform(55, 56, 50),
        "lon": rng.uniform(37, 38, 50),
        "y": rng.normal(100, 10, 50),
    })
    df.loc[df.index[:15], "y"] = np.nan

    result = profile_fit_transform(
        MeanImputer(strategy="median"),
        df_train=df, df_predict=df,
        lat="lat", lon="lon", target="y",
        method_name="mean_test",
    )
    assert isinstance(result, ProfileResult)
    assert result.method_name == "mean_test"
    assert result.n_train == 35  
    assert result.n_predict == 15
    assert result.fit_time_sec >= 0
    assert result.transform_time_sec >= 0
    assert result.fit_memory_peak_mb >= 0


def test_profile_fit_transform_to_dict():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "lat": rng.uniform(55, 56, 30),
        "lon": rng.uniform(37, 38, 30),
        "y": rng.normal(100, 10, 30),
    })
    df.loc[df.index[:10], "y"] = np.nan
    result = profile_fit_transform(
        MeanImputer(strategy="mean"),
        df_train=df, df_predict=df,
        lat="lat", lon="lon", target="y",
    )
    d = result.to_dict()
    assert d["method"] == "MeanImputer"
    assert "throughput_predict_per_sec" in d


def test_profile_scaling_runs_multiple_sizes():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "lat": rng.uniform(55, 56, 1500),
        "lon": rng.uniform(37, 38, 1500),
        "y": rng.normal(100, 10, 1500),
    })
    df_scaling = profile_scaling(
        lambda: IDWImputer(power=2, n_neighbors=5),
        df, lat="lat", lon="lon", target="y",
        sizes=[100, 500, 1000],
        mask_ratio=0.30,
        method_name="idw_scaling",
    )
    assert len(df_scaling) == 3
    assert "fit_time_sec" in df_scaling.columns
    assert "transform_time_sec" in df_scaling.columns
    assert df_scaling["error"].isna().all()
