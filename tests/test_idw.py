import numpy as np
import pandas as pd
import pytest

from spatialaug import IDWImputer


@pytest.fixture
def df_with_missing():
    lats = np.linspace(55.70, 55.80, 5)
    lons = np.linspace(37.55, 37.70, 5)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    grid_lat = grid_lat.flatten()
    grid_lon = grid_lon.flatten()
    rng = np.random.default_rng(42)
    price = 5_000_000 + 10_000_000 * (grid_lat - 55.70) + rng.normal(0, 100_000, len(grid_lat))
    df = pd.DataFrame({"lat": grid_lat, "lon": grid_lon, "price": price})
    df.loc[12, "price"] = np.nan
    return df


def test_idw_basic(df_with_missing):
    imp = IDWImputer(power=2, n_neighbors=4)
    result = imp.fit_transform(df_with_missing, lat="lat", lon="lon", target="price")
    assert result["price"].isna().sum() == 0
    predicted = result.loc[12, "price"]
    assert 5_000_000 < predicted < 7_500_000


def test_idw_invalid_power():
    with pytest.raises(ValueError, match="power must be positive"):
        IDWImputer(power=0)
    with pytest.raises(ValueError, match="power must be positive"):
        IDWImputer(power=-1)


def test_idw_invalid_n_neighbors():
    with pytest.raises(ValueError, match="n_neighbors must be >= 1"):
        IDWImputer(n_neighbors=0)


def test_idw_too_few_points():
    df = pd.DataFrame({"lat": [55.7, 55.8], "lon": [37.6, 37.7], "price": [5e6, np.nan]})
    imp = IDWImputer(n_neighbors=10)
    with pytest.raises(ValueError, match="Need at least 10 observed"):
        imp.fit(df, lat="lat", lon="lon", target="price")


def test_idw_k_one(df_with_missing):
    imp = IDWImputer(power=2, n_neighbors=1)
    result = imp.fit_transform(df_with_missing, lat="lat", lon="lon", target="price")
    assert result["price"].isna().sum() == 0


def test_idw_higher_power_more_local(df_with_missing):
    imp_low = IDWImputer(power=1, n_neighbors=8).fit(
        df_with_missing, lat="lat", lon="lon", target="price"
    )
    imp_high = IDWImputer(power=4, n_neighbors=8).fit(
        df_with_missing, lat="lat", lon="lon", target="price"
    )
    pred_low = imp_low.transform(df_with_missing).loc[12, "price"]
    pred_high = imp_high.transform(df_with_missing).loc[12, "price"]
    assert pred_low != pred_high


def test_idw_no_missing_returns_copy(df_with_missing):
    clean = df_with_missing.dropna().reset_index(drop=True)
    imp = IDWImputer(n_neighbors=3).fit(clean, lat="lat", lon="lon", target="price")
    result = imp.transform(clean)
    pd.testing.assert_frame_equal(result, clean)


def test_idw_transform_without_fit():
    imp = IDWImputer()
    with pytest.raises(RuntimeError, match="must be fitted"):
        imp.transform(pd.DataFrame({"lat": [0.0], "lon": [0.0], "price": [np.nan]}))
