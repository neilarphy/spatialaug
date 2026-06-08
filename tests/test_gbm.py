import numpy as np
import pandas as pd
import pytest

from spatialaug import GBMImputer


@pytest.fixture
def df_with_features():
    rng = np.random.default_rng(0)
    n = 100
    lat = 55.70 + 0.1 * rng.random(n)
    lon = 37.55 + 0.15 * rng.random(n)
    area = 30 + 100 * rng.random(n)
    price = 1_000_000 + 50_000 * area + 10_000_000 * (lat - 55.70) + rng.normal(0, 200_000, n)
    df = pd.DataFrame({"lat": lat, "lon": lon, "area": area, "price": price})
    df.loc[[10, 20, 30, 40, 50, 60, 70, 80, 90], "price"] = np.nan
    return df


def test_gbm_basic(df_with_features):
    imp = GBMImputer(n_estimators=30)
    result = imp.fit_transform(df_with_features, lat="lat", lon="lon", target="price")
    assert result["price"].isna().sum() == 0


def test_gbm_with_features(df_with_features):
    imp = GBMImputer(n_estimators=30, feature_cols=["area"])
    result = imp.fit_transform(df_with_features, lat="lat", lon="lon", target="price")
    assert result["price"].isna().sum() == 0
    assert imp._used_features_ == ["lat", "lon", "area"]


def test_gbm_invalid_n_estimators():
    with pytest.raises(ValueError, match="n_estimators must be"):
        GBMImputer(n_estimators=0)


def test_gbm_invalid_log_transform():
    with pytest.raises(ValueError, match="log_transform must be"):
        GBMImputer(log_transform="maybe")


def test_gbm_too_few_points():
    df = pd.DataFrame({"lat": [55.7] * 5, "lon": [37.6] * 5, "price": [1, 2, 3, 4, np.nan]})
    imp = GBMImputer()
    with pytest.raises(ValueError, match="at least 10"):
        imp.fit(df, lat="lat", lon="lon", target="price")


def test_gbm_missing_feature_col(df_with_features):
    imp = GBMImputer(feature_cols=["does_not_exist"])
    with pytest.raises(KeyError, match="feature_cols not in dataframe"):
        imp.fit(df_with_features, lat="lat", lon="lon", target="price")


def test_gbm_transform_without_fit():
    imp = GBMImputer()
    with pytest.raises(RuntimeError, match="must be fitted"):
        imp.transform(pd.DataFrame({"lat": [0.0], "lon": [0.0], "price": [np.nan]}))


def test_gbm_log_requires_positive():
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame(
        {
            "lat": 55 + rng.random(n),
            "lon": 37 + rng.random(n),
            "price": rng.normal(0, 1, n), 
        }
    )
    df.loc[0, "price"] = -100
    imp = GBMImputer(log_transform=True)
    with pytest.raises(ValueError, match="positive values"):
        imp.fit(df, lat="lat", lon="lon", target="price")
