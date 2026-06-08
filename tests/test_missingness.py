import numpy as np
import pandas as pd
import pytest

from spatialaug.benchmark import MissingnessSimulator


@pytest.fixture
def df_uniform():
    rng = np.random.default_rng(0)
    n = 200
    lat = 55.70 + 0.1 * rng.random(n)
    lon = 37.55 + 0.15 * rng.random(n)
    price = 1_000_000 + 10_000_000 * rng.random(n)
    return pd.DataFrame({"lat": lat, "lon": lon, "price": price})


def test_invalid_mechanism():
    with pytest.raises(ValueError, match="mechanism must be"):
        MissingnessSimulator(mechanism="bogus")


def test_invalid_ratio():
    with pytest.raises(ValueError, match="ratio must be"):
        MissingnessSimulator(ratio=0)
    with pytest.raises(ValueError, match="ratio must be"):
        MissingnessSimulator(ratio=1)


def test_mcar_correct_fraction(df_uniform):
    sim = MissingnessSimulator(mechanism="mcar", ratio=0.3, random_state=42)
    df_masked, mask = sim.apply(df_uniform, target_col="price")
    expected = int(len(df_uniform) * 0.3)
    assert mask.sum() == expected
    assert df_masked["price"].isna().sum() == expected


def test_mcar_reproducible(df_uniform):
    sim1 = MissingnessSimulator(mechanism="mcar", ratio=0.3, random_state=1)
    sim2 = MissingnessSimulator(mechanism="mcar", ratio=0.3, random_state=1)
    _, mask1 = sim1.apply(df_uniform, target_col="price")
    _, mask2 = sim2.apply(df_uniform, target_col="price")
    np.testing.assert_array_equal(mask1, mask2)


def test_diffuse_mnar_requires_coords(df_uniform):
    sim = MissingnessSimulator(mechanism="diffuse_mnar", ratio=0.3)
    with pytest.raises(ValueError, match="requires lat_col and lon_col"):
        sim.apply(df_uniform, target_col="price")


def test_diffuse_mnar_spatially_clustered(df_uniform):
    sim = MissingnessSimulator(
        mechanism="diffuse_mnar", ratio=0.3, random_state=42, n_clusters=3
    )
    _, mask = sim.apply(df_uniform, target_col="price", lat_col="lat", lon_col="lon")

    masked_coords = df_uniform.loc[mask, ["lat", "lon"]]
    blocked_var = masked_coords["lat"].var() + masked_coords["lon"].var()

    rng = np.random.default_rng(0)
    rand_idx = rng.choice(len(df_uniform), size=mask.sum(), replace=False)
    rand_coords = df_uniform.iloc[rand_idx][["lat", "lon"]]
    rand_var = rand_coords["lat"].var() + rand_coords["lon"].var()

    assert blocked_var < rand_var


def test_focused_mnar_high_tail_default_pool(df_uniform):
    sim = MissingnessSimulator(
        mechanism="focused_mnar", ratio=0.1, tail="high", focused_pool_factor=2.0
    )
    _, mask = sim.apply(df_uniform, target_col="price")
    masked_values = df_uniform.loc[mask, "price"]
    p80 = df_uniform["price"].quantile(0.8)
    assert (masked_values >= p80).all()


def test_focused_mnar_low_tail_default_pool(df_uniform):
    sim = MissingnessSimulator(
        mechanism="focused_mnar", ratio=0.1, tail="low", focused_pool_factor=2.0
    )
    _, mask = sim.apply(df_uniform, target_col="price")
    masked_values = df_uniform.loc[mask, "price"]
    p20 = df_uniform["price"].quantile(0.2)
    assert (masked_values <= p20).all()


def test_focused_mnar_deterministic_when_pool_factor_1(df_uniform):
    sim1 = MissingnessSimulator(
        mechanism="focused_mnar", ratio=0.1, focused_pool_factor=1.0, random_state=1
    )
    sim2 = MissingnessSimulator(
        mechanism="focused_mnar", ratio=0.1, focused_pool_factor=1.0, random_state=999
    )
    _, mask1 = sim1.apply(df_uniform, target_col="price")
    _, mask2 = sim2.apply(df_uniform, target_col="price")
    np.testing.assert_array_equal(mask1, mask2)
    p90 = df_uniform["price"].quantile(0.9)
    assert (df_uniform.loc[mask1, "price"] >= p90).all()


def test_focused_mnar_seed_varies_with_pool_factor_2(df_uniform):
    sim1 = MissingnessSimulator(
        mechanism="focused_mnar", ratio=0.1, focused_pool_factor=2.0, random_state=1
    )
    sim2 = MissingnessSimulator(
        mechanism="focused_mnar", ratio=0.1, focused_pool_factor=2.0, random_state=42
    )
    _, mask1 = sim1.apply(df_uniform, target_col="price")
    _, mask2 = sim2.apply(df_uniform, target_col="price")
    assert not np.array_equal(mask1, mask2)
    assert mask1.sum() == mask2.sum()


def test_focused_mnar_bias_preserved_with_random_pool(df_uniform):
    sim = MissingnessSimulator(
        mechanism="focused_mnar", ratio=0.3, tail="high",
        focused_pool_factor=2.0, random_state=42,
    )
    _, mask = sim.apply(df_uniform, target_col="price")
    masked_mean = df_uniform.loc[mask, "price"].mean()
    overall_mean = df_uniform["price"].mean()
    assert masked_mean > overall_mean


def test_focused_mnar_invalid_pool_factor():
    with pytest.raises(ValueError, match="focused_pool_factor must be >= 1.0"):
        MissingnessSimulator(focused_pool_factor=0.5)


def test_invalid_target_column(df_uniform):
    sim = MissingnessSimulator()
    with pytest.raises(KeyError, match="not in dataframe"):
        sim.apply(df_uniform, target_col="does_not_exist")
