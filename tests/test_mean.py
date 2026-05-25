"""Тесты MeanImputer."""

import numpy as np
import pandas as pd
import pytest

from spatialaug import MeanImputer


@pytest.fixture
def df_with_missing():
    """Маленький датафрейм: 2 региона по 3 точки, по 1 пропуску в каждом."""
    return pd.DataFrame(
        {
            "geo_lat": [55.7, 55.8, 55.9, 59.9, 59.95, 59.85],
            "geo_lon": [37.6, 37.5, 37.7, 30.3, 30.35, 30.25],
            "region": ["msk", "msk", "msk", "spb", "spb", "spb"],
            "price": [5_000_000, np.nan, 6_000_000, 8_000_000, np.nan, 9_000_000],
        }
    )


def test_mean_imputer_no_group(df_with_missing):
    imp = MeanImputer(strategy="mean")
    result = imp.fit_transform(
        df_with_missing, lat="geo_lat", lon="geo_lon", target="price"
    )
    # Глобальное среднее по наблюдаемым = (5M+6M+8M+9M)/4 = 7M
    assert result["price"].isna().sum() == 0
    assert result.loc[1, "price"] == pytest.approx(7_000_000)
    assert result.loc[4, "price"] == pytest.approx(7_000_000)


def test_mean_imputer_with_group(df_with_missing):
    imp = MeanImputer(strategy="mean")
    result = imp.fit_transform(
        df_with_missing,
        lat="geo_lat",
        lon="geo_lon",
        target="price",
        group="region",
    )
    # msk mean = (5M+6M)/2 = 5.5M; spb mean = (8M+9M)/2 = 8.5M
    assert result.loc[1, "price"] == pytest.approx(5_500_000)
    assert result.loc[4, "price"] == pytest.approx(8_500_000)


def test_mean_imputer_median(df_with_missing):
    imp = MeanImputer(strategy="median")
    result = imp.fit_transform(
        df_with_missing,
        lat="geo_lat",
        lon="geo_lon",
        target="price",
        group="region",
    )
    # Из 2 точек медиана = mean, поэтому те же 5.5M / 8.5M
    assert result.loc[1, "price"] == pytest.approx(5_500_000)
    assert result.loc[4, "price"] == pytest.approx(8_500_000)


def test_unseen_group_falls_back_to_global(df_with_missing):
    imp = MeanImputer(strategy="mean")
    imp.fit(
        df_with_missing,
        lat="geo_lat",
        lon="geo_lon",
        target="price",
        group="region",
    )

    new_df = pd.DataFrame(
        {
            "geo_lat": [56.0],
            "geo_lon": [40.0],
            "region": ["nsk"],  # неизвестный регион
            "price": [np.nan],
        }
    )
    result = imp.transform(new_df)
    # Глобальное среднее = 7M (см. test_mean_imputer_no_group)
    assert result.loc[0, "price"] == pytest.approx(7_000_000)


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="strategy must be"):
        MeanImputer(strategy="mode")


def test_transform_without_fit_raises():
    imp = MeanImputer()
    with pytest.raises(RuntimeError, match="must be fitted"):
        imp.transform(pd.DataFrame({"price": [1.0]}))


def test_no_missing_returns_copy(df_with_missing):
    """Если NaN нет — возвращаем копию без изменений."""
    clean = df_with_missing.dropna().reset_index(drop=True)
    imp = MeanImputer().fit(clean, lat="geo_lat", lon="geo_lon", target="price")
    result = imp.transform(clean)
    pd.testing.assert_frame_equal(result, clean)


def test_repr():
    imp = MeanImputer()
    assert "not fitted" in repr(imp)
    imp.is_fitted = True
    assert "fitted" in repr(imp)
