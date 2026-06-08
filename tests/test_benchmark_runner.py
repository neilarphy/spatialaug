import numpy as np
import pandas as pd
import pytest

from spatialaug import IDWImputer, MeanImputer
from spatialaug.benchmark import run_benchmark


@pytest.fixture
def df_simple():
    rng = np.random.default_rng(0)
    n = 100
    lat = 55.70 + 0.1 * rng.random(n)
    lon = 37.55 + 0.15 * rng.random(n)
    price = 5_000_000 + 10_000_000 * (lat - 55.70) + rng.normal(0, 100_000, n)
    return pd.DataFrame({"lat": lat, "lon": lon, "price": price})


def test_run_benchmark_basic_schema(df_simple):
    results = run_benchmark(
        df_simple,
        target="price",
        lat="lat",
        lon="lon",
        imputer_factories={
            "mean": lambda: MeanImputer(strategy="median"),
            "idw": lambda: IDWImputer(power=2, n_neighbors=5),
        },
        mechanisms=("mcar",),
        verbose=False,
    )
    expected_cols = {
        "imputer",
        "mechanism",
        "n_missing",
        "missingness_ratio",
        "mae",
        "rmse",
        "fit_time_sec",
        "transform_time_sec",
        "error",
    }
    assert expected_cols.issubset(results.columns)
    assert len(results) == 2


def test_run_benchmark_multiple_mechanisms(df_simple):
    results = run_benchmark(
        df_simple,
        target="price",
        lat="lat",
        lon="lon",
        imputer_factories={"mean": lambda: MeanImputer()},
        mechanisms=("mcar", "diffuse_mnar", "focused_mnar"),
        verbose=False,
    )
    assert len(results) == 3
    assert set(results["mechanism"]) == {"mcar", "diffuse_mnar", "focused_mnar"}


def test_run_benchmark_error_handling(df_simple):
    class BrokenImputer:
        def fit(self, *args, **kwargs):
            raise RuntimeError("nope")

    results = run_benchmark(
        df_simple,
        target="price",
        lat="lat",
        lon="lon",
        imputer_factories={
            "ok": lambda: MeanImputer(),
            "broken": lambda: BrokenImputer(),
        },
        mechanisms=("mcar",),
        verbose=False,
    )
    assert len(results) == 2
    broken_row = results[results["imputer"] == "broken"].iloc[0]
    assert pd.isna(broken_row["mae"])
    assert "nope" in broken_row["error"]


def test_run_benchmark_metrics_finite(df_simple):
    results = run_benchmark(
        df_simple,
        target="price",
        lat="lat",
        lon="lon",
        imputer_factories={
            "mean": lambda: MeanImputer(),
            "idw": lambda: IDWImputer(power=2, n_neighbors=5),
        },
        mechanisms=("mcar",),
        verbose=False,
    )
    assert results["mae"].notna().all()
    assert results["rmse"].notna().all()
    assert (results["mae"] >= 0).all()
    assert (results["rmse"] >= 0).all()
