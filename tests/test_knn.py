import numpy as np
import pandas as pd
import pytest

from spatialaug import KNNImputer


@pytest.fixture
def df_with_missing():
    lats = np.linspace(55.70, 55.80, 5)
    lons = np.linspace(37.55, 37.70, 5)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    grid_lat = grid_lat.flatten()
    grid_lon = grid_lon.flatten()
    rng = np.random.default_rng(0)
    price = 5_000_000 + 10_000_000 * (grid_lat - 55.70) + rng.normal(0, 50_000, len(grid_lat))
    df = pd.DataFrame({"lat": grid_lat, "lon": grid_lon, "price": price})
    df.loc[12, "price"] = np.nan
    return df


def test_knn_mean(df_with_missing):
    imp = KNNImputer(n_neighbors=4, aggregation="mean")
    result = imp.fit_transform(df_with_missing, lat="lat", lon="lon", target="price")
    assert result["price"].isna().sum() == 0


def test_knn_median(df_with_missing):
    imp = KNNImputer(n_neighbors=4, aggregation="median")
    result = imp.fit_transform(df_with_missing, lat="lat", lon="lon", target="price")
    assert result["price"].isna().sum() == 0


def test_knn_invalid_aggregation():
    with pytest.raises(ValueError, match="aggregation must be"):
        KNNImputer(aggregation="mode")


def test_knn_invalid_n_neighbors():
    with pytest.raises(ValueError, match="n_neighbors must be >= 1"):
        KNNImputer(n_neighbors=0)


def test_knn_k_one_returns_nearest(df_with_missing):
    imp = KNNImputer(n_neighbors=1)
    result = imp.fit_transform(df_with_missing, lat="lat", lon="lon", target="price")
    assert not np.isnan(result.loc[12, "price"])


def test_knn_too_few_points():
    df = pd.DataFrame({"lat": [55.7, 55.8], "lon": [37.6, 37.7], "price": [5e6, np.nan]})
    imp = KNNImputer(n_neighbors=10)
    with pytest.raises(ValueError, match="Need at least 10 observed"):
        imp.fit(df, lat="lat", lon="lon", target="price")


def test_knn_no_missing_returns_copy(df_with_missing):
    clean = df_with_missing.dropna().reset_index(drop=True)
    imp = KNNImputer(n_neighbors=3).fit(clean, lat="lat", lon="lon", target="price")
    result = imp.transform(clean)
    pd.testing.assert_frame_equal(result, clean)


def test_knn_transform_without_fit():
    imp = KNNImputer()
    with pytest.raises(RuntimeError, match="must be fitted"):
        imp.transform(pd.DataFrame({"lat": [0.0], "lon": [0.0], "price": [np.nan]}))
