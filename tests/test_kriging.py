import numpy as np
import pandas as pd
import pytest

from spatialaug import KrigingImputer


@pytest.fixture
def df_smooth_field():
    rng = np.random.default_rng(42)
    n = 60
    lat = 55.70 + 0.1 * rng.random(n)
    lon = 37.55 + 0.15 * rng.random(n)
    z = 100 + 50 * lat + 30 * lon + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"lat": lat, "lon": lon, "z": z})
    df.loc[[5, 15, 25, 35, 45], "z"] = np.nan
    return df


@pytest.fixture
def df_skewed_field():
    rng = np.random.default_rng(0)
    n = 60
    lat = 55.70 + 0.1 * rng.random(n)
    lon = 37.55 + 0.15 * rng.random(n)
    log_price = 14 + 5 * (lat - 55.70) + 2 * (lon - 37.55) + rng.normal(0, 0.7, n)
    price = np.exp(log_price)
    df = pd.DataFrame({"lat": lat, "lon": lon, "price": price})
    df.loc[[5, 15, 25, 35, 45], "price"] = np.nan
    return df


def test_invalid_variogram_model():
    with pytest.raises(ValueError, match="variogram_model must be"):
        KrigingImputer(variogram_model="bogus")


def test_invalid_detrend():
    with pytest.raises(ValueError, match="detrend must be"):
        KrigingImputer(detrend="cubic")


def test_invalid_log_transform():
    with pytest.raises(ValueError, match="log_transform must be"):
        KrigingImputer(log_transform="maybe")


def test_invalid_n_neighbors():
    with pytest.raises(ValueError, match="n_neighbors must be"):
        KrigingImputer(n_neighbors=0)


@pytest.mark.slow
def test_kriging_smooth_field(df_smooth_field):
    imp = KrigingImputer(
        variogram_model="spherical",
        log_transform=False,
        detrend="linear",
    )
    result = imp.fit_transform(df_smooth_field, lat="lat", lon="lon", target="z")
    assert result["z"].isna().sum() == 0
    observed_min = df_smooth_field["z"].dropna().min()
    observed_max = df_smooth_field["z"].dropna().max()
    span = observed_max - observed_min
    assert result["z"].between(observed_min - 0.05 * span, observed_max + 0.05 * span).all()


@pytest.mark.slow
def test_kriging_auto_log_transform(df_skewed_field):
    imp = KrigingImputer(log_transform="auto", detrend="linear")
    imp.fit(df_skewed_field, lat="lat", lon="lon", target="price")
    assert imp.log_applied_ is True


@pytest.mark.slow
def test_kriging_auto_no_log_smooth(df_smooth_field):
    imp = KrigingImputer(log_transform="auto", detrend="linear")
    imp.fit(df_smooth_field, lat="lat", lon="lon", target="z")
    assert imp.log_applied_ is False


@pytest.mark.slow
def test_kriging_log_requires_positive(df_smooth_field):
    df = df_smooth_field.copy()
    df.loc[0, "z"] = -100
    imp = KrigingImputer(log_transform=True)
    with pytest.raises(ValueError, match="positive values"):
        imp.fit(df, lat="lat", lon="lon", target="z")


@pytest.mark.slow
def test_kriging_no_detrend(df_smooth_field):
    imp = KrigingImputer(detrend=None, log_transform=False)
    result = imp.fit_transform(df_smooth_field, lat="lat", lon="lon", target="z")
    assert result["z"].isna().sum() == 0


@pytest.mark.slow
def test_kriging_quadratic_detrend(df_smooth_field):
    imp = KrigingImputer(detrend="quadratic", log_transform=False)
    result = imp.fit_transform(df_smooth_field, lat="lat", lon="lon", target="z")
    assert result["z"].isna().sum() == 0


@pytest.mark.slow
def test_kriging_local_mode(df_smooth_field):
    imp = KrigingImputer(
        variogram_model="spherical",
        log_transform=False,
        detrend="linear",
        local=True,
        n_neighbors=20,
    )
    result = imp.fit_transform(df_smooth_field, lat="lat", lon="lon", target="z")
    assert result["z"].isna().sum() == 0


@pytest.mark.slow
def test_kriging_too_few_points():
    df = pd.DataFrame(
        {"lat": [55.7, 55.8, 55.9], "lon": [37.6, 37.7, 37.8], "z": [1.0, 2.0, np.nan]}
    )
    imp = KrigingImputer()
    with pytest.raises(ValueError, match="at least 5 observed"):
        imp.fit(df, lat="lat", lon="lon", target="z")


def test_kriging_transform_without_fit():
    imp = KrigingImputer()
    with pytest.raises(RuntimeError, match="must be fitted"):
        imp.transform(pd.DataFrame({"lat": [0.0], "lon": [0.0], "z": [np.nan]}))
