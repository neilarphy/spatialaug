import numpy as np
import pandas as pd
import pytest

from spatialaug import KrigingImputer, MeanImputer
from spatialaug.augmenters import Augmenter


@pytest.fixture
def synthetic_df():
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame({
        "lat": rng.uniform(55, 56, n),
        "lon": rng.uniform(37, 38, n),
        "kkt_count": rng.poisson(50, n) + 1,
        "is_mall": rng.choice([0, 1], n).astype(int),
        "avg_bill": rng.lognormal(7, 0.5, n),
    })


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        Augmenter(MeanImputer(), strategy="invalid")


def test_regular_grid_creates_synthetic_points(synthetic_df):
    aug = Augmenter(
        MeanImputer(strategy="median"), strategy="regular_grid",
        n_synthetic=0.5, grid_step_km=2.0,
    )
    out = aug.fit_transform(
        synthetic_df, lat="lat", lon="lon", target="avg_bill",
        feature_cols=["kkt_count", "is_mall"],
    )
    assert "_is_synthetic" in out.columns
    n_synth = out["_is_synthetic"].sum()
    assert n_synth > 0
    assert (~out["_is_synthetic"]).sum() == len(synthetic_df)


def test_jitter_perturbs_existing_coordinates(synthetic_df):
    aug = Augmenter(
        MeanImputer(strategy="median"), strategy="jitter",
        n_synthetic=20, jitter_km=0.3, random_state=1,
    )
    out = aug.fit_transform(
        synthetic_df, lat="lat", lon="lon", target="avg_bill",
        feature_cols=["kkt_count"],
    )
    synth = out[out["_is_synthetic"]]
    assert len(synth) > 0
    obs_set = set(zip(synthetic_df.lat.round(6), synthetic_df.lon.round(6)))
    synth_set = set(zip(synth.lat.round(6), synth.lon.round(6)))
    overlap = obs_set & synth_set
    assert len(overlap) == 0 


def test_density_fill_targets_sparse_zones(synthetic_df):
    aug = Augmenter(
        MeanImputer(strategy="median"), strategy="density_fill",
        n_synthetic=20, random_state=2,
    )
    out = aug.fit_transform(
        synthetic_df, lat="lat", lon="lon", target="avg_bill",
        feature_cols=["kkt_count"],
    )
    synth = out[out["_is_synthetic"]]
    assert len(synth) > 0


def test_mixup_blends_pairs(synthetic_df):
    aug = Augmenter(
        MeanImputer(strategy="median"), strategy="mixup",
        n_synthetic=30, random_state=3,
    )
    out = aug.fit_transform(
        synthetic_df, lat="lat", lon="lon", target="avg_bill",
        feature_cols=["kkt_count", "is_mall"],
    )
    synth = out[out["_is_synthetic"]]
    assert len(synth) > 0
    assert synth.lat.min() >= synthetic_df.lat.min()
    assert synth.lat.max() <= synthetic_df.lat.max()


def test_exclude_df_filters_synthetic_near_test(synthetic_df):
    test_df = synthetic_df.iloc[:30].copy()
    train_df = synthetic_df.iloc[30:].copy()
    aug = Augmenter(
        MeanImputer(strategy="median"), strategy="regular_grid",
        n_synthetic=20, min_distance_km=2.0, random_state=4,
    )
    out = aug.fit_transform(
        train_df, lat="lat", lon="lon", target="avg_bill",
        feature_cols=["kkt_count"],
        exclude_df=test_df,
    )
    synth = out[out["_is_synthetic"]]
    if len(synth) > 0:
        test_set = set(zip(test_df.lat.round(6), test_df.lon.round(6)))
        synth_set = set(zip(synth.lat.round(6), synth.lon.round(6)))
        assert len(test_set & synth_set) == 0


def test_n_synthetic_as_fraction(synthetic_df):
    aug = Augmenter(
        MeanImputer(strategy="median"), strategy="regular_grid",
        n_synthetic=0.5, 
    )
    out = aug.fit_transform(
        synthetic_df, lat="lat", lon="lon", target="avg_bill",
    )
    n_synth = out["_is_synthetic"].sum()
    assert n_synth <= len(synthetic_df) * 0.6
