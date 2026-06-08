"""Тесты UniversalKrigingImputer."""

import numpy as np
import pandas as pd
import pytest

from spatialaug import UniversalKrigingImputer


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


def test_invalid_variogram_model():
    with pytest.raises(ValueError, match="variogram_model must be"):
        UniversalKrigingImputer(variogram_model="bogus")


def test_invalid_drift_terms():
    with pytest.raises(ValueError, match="Unsupported drift_terms"):
        UniversalKrigingImputer(drift_terms=("external_Z",))


def test_invalid_log_transform():
    with pytest.raises(ValueError, match="log_transform must be"):
        UniversalKrigingImputer(log_transform="maybe")


@pytest.mark.slow
def test_uk_smooth_field(df_smooth_field):
    imp = UniversalKrigingImputer(variogram_model="spherical", log_transform=False)
    result = imp.fit_transform(df_smooth_field, lat="lat", lon="lon", target="z")
    assert result["z"].isna().sum() == 0
    observed_min = df_smooth_field["z"].dropna().min()
    observed_max = df_smooth_field["z"].dropna().max()
    span = observed_max - observed_min
    assert result["z"].between(
        observed_min - 0.05 * span, observed_max + 0.05 * span
    ).all()


@pytest.mark.slow
def test_uk_auto_log_transform():
    rng = np.random.default_rng(0)
    n = 60
    lat = 55.70 + 0.1 * rng.random(n)
    lon = 37.55 + 0.15 * rng.random(n)
    log_price = 14 + 5 * (lat - 55.70) + rng.normal(0, 0.7, n)
    price = np.exp(log_price)
    df = pd.DataFrame({"lat": lat, "lon": lon, "price": price})
    df.loc[[5, 15, 25, 35, 45], "price"] = np.nan

    imp = UniversalKrigingImputer(log_transform="auto")
    imp.fit(df, lat="lat", lon="lon", target="price")
    assert imp.log_applied_ is True


def test_uk_too_few_points():
    df = pd.DataFrame(
        {"lat": [55.7, 55.8], "lon": [37.6, 37.7], "z": [1.0, np.nan]}
    )
    imp = UniversalKrigingImputer()
    with pytest.raises(ValueError, match="at least 5"):
        imp.fit(df, lat="lat", lon="lon", target="z")


def test_uk_transform_without_fit():
    imp = UniversalKrigingImputer()
    with pytest.raises(RuntimeError, match="must be fitted"):
        imp.transform(pd.DataFrame({"lat": [0.0], "lon": [0.0], "z": [np.nan]}))




@pytest.fixture
def df_ked_field():
    rng = np.random.default_rng(123)
    n = 80
    lat = 55.7 + 0.1 * rng.random(n)
    lon = 37.55 + 0.15 * rng.random(n)
    x_feat = rng.uniform(0, 20, n)
    z = 100 + 50 * lat + 30 * lon + 10 * x_feat + rng.normal(0, 1.0, n)
    df = pd.DataFrame({"lat": lat, "lon": lon, "x_feat": x_feat, "z": z})
    df.loc[[5, 15, 25, 35, 45, 55, 65, 75], "z"] = np.nan
    return df


def test_ked_accepts_feature_cols():
    imp = UniversalKrigingImputer(feature_cols=["x_feat"])
    assert imp.feature_cols == ("x_feat",)
    assert imp.drift_terms == ("specified",)


def test_ked_empty_feature_cols_raises():
    with pytest.raises(ValueError, match="must contain at least one"):
        UniversalKrigingImputer(feature_cols=[])


def test_ked_missing_feature_col_raises(df_ked_field):
    imp = UniversalKrigingImputer(feature_cols=["nonexistent_col"], log_transform=False)
    with pytest.raises(ValueError, match="feature_cols missing"):
        imp.fit(df_ked_field, lat="lat", lon="lon", target="z")


@pytest.mark.slow
def test_ked_recovers_drift_field(df_ked_field):
    plain = UniversalKrigingImputer(log_transform=False)
    ked = UniversalKrigingImputer(feature_cols=["x_feat"], log_transform=False)

    plain_pred = plain.fit_transform(df_ked_field, lat="lat", lon="lon", target="z")
    ked_pred = ked.fit_transform(df_ked_field, lat="lat", lon="lon", target="z")

    rng = np.random.default_rng(123)
    n = 80
    lat = 55.7 + 0.1 * rng.random(n)
    lon = 37.55 + 0.15 * rng.random(n)
    x_feat = rng.uniform(0, 20, n)
    true_z = 100 + 50 * lat + 30 * lon + 10 * x_feat + rng.normal(0, 1.0, n)
    mask_idx = [5, 15, 25, 35, 45, 55, 65, 75]
    true_at_mask = true_z[mask_idx]

    plain_at_mask = plain_pred.loc[mask_idx, "z"].to_numpy()
    ked_at_mask = ked_pred.loc[mask_idx, "z"].to_numpy()

    plain_mae = np.mean(np.abs(plain_at_mask - true_at_mask))
    ked_mae = np.mean(np.abs(ked_at_mask - true_at_mask))

    assert ked_mae < plain_mae, (
        f"KED MAE {ked_mae:.2f} should be < plain UK MAE {plain_mae:.2f}"
    )


def test_ked_nan_in_query_features_raises(df_ked_field):
    df = df_ked_field.copy()
    imp = UniversalKrigingImputer(feature_cols=["x_feat"], log_transform=False)
    imp.fit(df, lat="lat", lon="lon", target="z")
    df.loc[5, "x_feat"] = np.nan
    with pytest.raises(ValueError, match="NaN in feature_cols at query points"):
        imp.transform(df)


def test_ked_drops_training_rows_with_nan_features():
    rng = np.random.default_rng(0)
    n = 30
    lat = 55.7 + 0.1 * rng.random(n)
    lon = 37.55 + 0.15 * rng.random(n)
    x_feat = rng.uniform(0, 20, n)
    z = 100 + 10 * x_feat + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"lat": lat, "lon": lon, "x_feat": x_feat, "z": z})
    df.loc[0, "x_feat"] = np.nan
    df.loc[[10, 20], "z"] = np.nan

    imp = UniversalKrigingImputer(feature_cols=["x_feat"], log_transform=False)
    imp.fit(df, lat="lat", lon="lon", target="z")
    assert imp.kriging_ is not None


def test_ked_standardize_features_default_on():
    df = pd.DataFrame({
        "lat": np.linspace(55.7, 55.8, 20),
        "lon": np.linspace(37.5, 37.6, 20),
        "x_feat": np.linspace(0, 1000, 20), 
        "z": np.linspace(100, 200, 20),
    })
    df.loc[[5, 10], "z"] = np.nan
    imp = UniversalKrigingImputer(feature_cols=["x_feat"], log_transform=False)
    imp.fit(df, lat="lat", lon="lon", target="z")
    assert imp.feature_means_ is not None
    assert imp.feature_stds_ is not None


def test_ked_converts_string_features_to_numeric():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "lat": rng.uniform(55.7, 55.8, n),
        "lon": rng.uniform(37.5, 37.6, n),
        "area": rng.uniform(30, 100, n),
        "rooms": [str(int(r)) for r in rng.integers(1, 5, n)],
        "z": rng.normal(100, 10, n),
    })
    df.loc[[5, 10, 15], "z"] = np.nan

    imp = UniversalKrigingImputer(
        feature_cols=["area", "rooms"], log_transform=False
    )
    result = imp.fit_transform(df, lat="lat", lon="lon", target="z")
    assert result["z"].notna().all()


def test_ked_log_inverse_is_stable_under_extrapolation():
    rng = np.random.default_rng(0)
    n_train = 30
    train = pd.DataFrame({
        "lat": rng.uniform(55.7, 55.8, n_train),
        "lon": rng.uniform(37.5, 37.6, n_train),
        "x_feat": rng.uniform(0, 10, n_train),
        "z": np.exp(10 + 0.5 * rng.uniform(0, 10, n_train)
                    + rng.normal(0, 0.3, n_train)),
    })
    query = pd.DataFrame({
        "lat": [55.75], "lon": [37.55], "x_feat": [50.0], "z": [np.nan],
    })
    combined = pd.concat([train, query], ignore_index=True)

    imp = UniversalKrigingImputer(feature_cols=["x_feat"], log_transform=True)
    result = imp.fit_transform(combined, lat="lat", lon="lon", target="z")
    pred = result.loc[result["z"].notna(), "z"].iloc[-1]  # last row = query
    assert np.isfinite(pred)
    assert pred < 1e10, f"KED prediction взорвалось: {pred}"


def test_ked_can_disable_standardization():
    df = pd.DataFrame({
        "lat": np.linspace(55.7, 55.8, 20),
        "lon": np.linspace(37.5, 37.6, 20),
        "x_feat": np.linspace(0, 1, 20), 
        "z": np.linspace(100, 200, 20),
    })
    df.loc[[5, 10], "z"] = np.nan
    imp = UniversalKrigingImputer(
        feature_cols=["x_feat"], standardize_features=False, log_transform=False
    )
    imp.fit(df, lat="lat", lon="lon", target="z")
    assert imp.feature_means_ is None
    assert imp.feature_stds_ is None
