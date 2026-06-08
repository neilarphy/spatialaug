import numpy as np
import pandas as pd

from spatialaug import MeanImputer
from spatialaug.benchmark import run_benchmark_multi_seed


def make_synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "lat": rng.uniform(55, 56, n),
        "lon": rng.uniform(37, 38, n),
        "y": rng.normal(100, 10, n),
    })


def test_multi_seed_returns_concat_df():
    df = make_synthetic_df(n=100)
    factories = {"mean": lambda: MeanImputer(strategy="median")}
    results = run_benchmark_multi_seed(
        df=df, target="y", lat="lat", lon="lon",
        imputer_factories=factories,
        mechanisms=("mcar",),
        missingness_ratio=0.3,
        seeds=(1, 2, 3),
        verbose=False,
    )
    assert len(results) == 3
    assert "seed" in results.columns
    assert sorted(results["seed"].unique().tolist()) == [1, 2, 3]


def test_multi_seed_different_seeds_give_different_results():
    df = make_synthetic_df(n=100)
    factories = {"mean": lambda: MeanImputer(strategy="mean")}
    results = run_benchmark_multi_seed(
        df=df, target="y", lat="lat", lon="lon",
        imputer_factories=factories,
        mechanisms=("mcar",),
        missingness_ratio=0.3,
        seeds=(1, 2, 3, 7, 42),
        verbose=False,
    )
    mae_values = results["mae"].to_numpy()
    assert mae_values.std() > 0


def test_multi_seed_preserves_all_columns():
    df = make_synthetic_df(n=100)
    factories = {"mean": lambda: MeanImputer(strategy="median")}
    results = run_benchmark_multi_seed(
        df=df, target="y", lat="lat", lon="lon",
        imputer_factories=factories,
        mechanisms=("mcar", "diffuse_mnar"),
        missingness_ratio=0.3,
        seeds=(1, 2),
        verbose=False,
    )
    for col in ["imputer", "mechanism", "mae", "rmse", "fit_time_sec",
                "transform_time_sec", "seed"]:
        assert col in results.columns


def test_multi_seed_aggregation_pattern():
    df = make_synthetic_df(n=300)
    factories = {"mean": lambda: MeanImputer(strategy="median")}
    results = run_benchmark_multi_seed(
        df=df, target="y", lat="lat", lon="lon",
        imputer_factories=factories,
        mechanisms=("mcar",),
        missingness_ratio=0.3,
        seeds=(1, 2, 3, 4, 5),
        verbose=False,
    )
    agg = results.groupby(["imputer", "mechanism"])["mae"].agg(["mean", "std"])
    assert "mean" in agg.columns
    assert "std" in agg.columns
    assert agg["std"].iloc[0] >= 0
    assert not np.isnan(agg["std"].iloc[0])
