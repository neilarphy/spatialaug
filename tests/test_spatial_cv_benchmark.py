import numpy as np
import pandas as pd

from spatialaug import IDWImputer, MeanImputer
from spatialaug.benchmark import run_benchmark_spatial_cv


def make_grid_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "lat": rng.uniform(55, 56, n),
        "lon": rng.uniform(37, 38, n),
        "y": rng.normal(100, 10, n),
    })


def test_spatial_cv_returns_per_fold_rows():
    df = make_grid_df(n=200)
    factories = {"mean": lambda: MeanImputer(strategy="median")}
    results = run_benchmark_spatial_cv(
        df=df, target="y", lat="lat", lon="lon",
        imputer_factories=factories, n_splits=5,
        verbose=False,
    )
    assert len(results) == 5
    assert sorted(results["fold"].unique()) == [0, 1, 2, 3, 4]
    assert (results["mechanism"] == "spatial_cv").all()


def test_spatial_cv_required_columns():
    df = make_grid_df(n=200)
    factories = {"mean": lambda: MeanImputer(strategy="median")}
    results = run_benchmark_spatial_cv(
        df=df, target="y", lat="lat", lon="lon",
        imputer_factories=factories, n_splits=3, verbose=False,
    )
    for col in ["imputer", "fold", "mechanism", "mae", "rmse",
                "fit_time_sec", "transform_time_sec", "n_test"]:
        assert col in results.columns


def test_spatial_cv_test_blocks_dont_overlap_with_train():
    df = make_grid_df(n=200)
    factories = {"idw": lambda: IDWImputer(power=2, n_neighbors=5)}
    results = run_benchmark_spatial_cv(
        df=df, target="y", lat="lat", lon="lon",
        imputer_factories=factories, n_splits=4, verbose=False,
    )
    assert results["n_test"].sum() == len(df)


def test_spatial_cv_handles_multiple_imputers():
    df = make_grid_df(n=200)
    factories = {
        "mean": lambda: MeanImputer(strategy="median"),
        "idw": lambda: IDWImputer(power=2, n_neighbors=5),
    }
    results = run_benchmark_spatial_cv(
        df=df, target="y", lat="lat", lon="lon",
        imputer_factories=factories, n_splits=3, verbose=False,
    )
    assert len(results) == 6
    assert set(results["imputer"].unique()) == {"mean", "idw"}


def test_spatial_cv_mae_finite():
    df = make_grid_df(n=200)
    factories = {"mean": lambda: MeanImputer(strategy="median")}
    results = run_benchmark_spatial_cv(
        df=df, target="y", lat="lat", lon="lon",
        imputer_factories=factories, n_splits=4, verbose=False,
    )
    assert results["mae"].notna().all()
    assert (results["mae"] > 0).all()
    assert (results["mae"] < 1000).all()  
