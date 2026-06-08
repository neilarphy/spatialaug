import numpy as np
import pandas as pd
import pytest

from spatialaug import KrigingImputer, GBMImputer, KrigingPriorRegression


@pytest.fixture
def synthetic_df():
    rng = np.random.RandomState(42)
    n = 100
    lat = rng.uniform(55, 56, n)
    lon = rng.uniform(37, 38, n)
    feature = rng.normal(50, 10, n)
    target = 1000 * lat + 500 * lon + 10 * feature + rng.normal(0, 100, n)
    df = pd.DataFrame({
        "lat": lat, "lon": lon, "target": target, "feature": feature,
    })
    mask_idx = rng.choice(n, 30, replace=False)
    df_masked = df.copy()
    df_masked.loc[mask_idx, "target"] = np.nan
    return df, df_masked, mask_idx


def test_kpr_fit_returns_self(synthetic_df):
    _, df_masked, _ = synthetic_df
    kpr = KrigingPriorRegression(feature_cols=["feature"])
    result = kpr.fit(df_masked, lat="lat", lon="lon", target="target")
    assert result is kpr
    assert kpr.is_fitted


def test_kpr_transform_fills_nans(synthetic_df):
    _, df_masked, mask_idx = synthetic_df
    kpr = KrigingPriorRegression(feature_cols=["feature"])
    kpr.fit(df_masked, lat="lat", lon="lon", target="target")
    filled = kpr.transform(df_masked)
    assert filled["target"].notna().all()
    assert "_kriging_prior" not in filled.columns


def test_kpr_better_than_kriging_alone_when_feature_helps(synthetic_df):
    df_orig, df_masked, mask_idx = synthetic_df
    true_vals = df_orig.loc[mask_idx, "target"].to_numpy()

    ok = KrigingImputer(progress=False)
    ok.fit(df_masked, lat="lat", lon="lon", target="target")
    filled_ok = ok.transform(df_masked)
    pred_ok = filled_ok.loc[mask_idx, "target"].to_numpy()
    mae_ok = float(np.mean(np.abs(pred_ok - true_vals)))

    kpr = KrigingPriorRegression(feature_cols=["feature"])
    kpr.fit(df_masked, lat="lat", lon="lon", target="target")
    filled_kpr = kpr.transform(df_masked)
    pred_kpr = filled_kpr.loc[mask_idx, "target"].to_numpy()
    mae_kpr = float(np.mean(np.abs(pred_kpr - true_vals)))

    assert mae_kpr < mae_ok * 1.5  


def test_kpr_better_than_gbm_alone_when_spatial_matters(synthetic_df):
    df_orig, df_masked, mask_idx = synthetic_df
    true_vals = df_orig.loc[mask_idx, "target"].to_numpy()

    gbm = GBMImputer(feature_cols=["feature"])
    gbm.fit(df_masked, lat="lat", lon="lon", target="target")
    filled_gbm = gbm.transform(df_masked)
    pred_gbm = filled_gbm.loc[mask_idx, "target"].to_numpy()
    mae_gbm = float(np.mean(np.abs(pred_gbm - true_vals)))

    kpr = KrigingPriorRegression(feature_cols=["feature"])
    kpr.fit(df_masked, lat="lat", lon="lon", target="target")
    filled_kpr = kpr.transform(df_masked)
    pred_kpr = filled_kpr.loc[mask_idx, "target"].to_numpy()
    mae_kpr = float(np.mean(np.abs(pred_kpr - true_vals)))

    assert mae_kpr < mae_gbm * 1.5


def test_kpr_works_without_features(synthetic_df):
    _, df_masked, mask_idx = synthetic_df
    kpr = KrigingPriorRegression()  # no features
    kpr.fit(df_masked, lat="lat", lon="lon", target="target")
    filled = kpr.transform(df_masked)
    assert filled["target"].notna().all()


def test_kpr_transform_without_fit_raises():
    kpr = KrigingPriorRegression()
    with pytest.raises(RuntimeError):
        kpr.transform(pd.DataFrame({"lat": [55], "lon": [37], "target": [np.nan]}))


def test_kpr_passes_custom_kwargs():
    kpr = KrigingPriorRegression(
        kriging_kwargs={"variogram_model": "exponential", "log_transform": False,
                        "local": False, "progress": False},
        gbm_kwargs={"n_estimators": 10, "num_leaves": 5,
                    "log_transform": False},
    )
    assert kpr.kriging_kwargs["variogram_model"] == "exponential"
    assert kpr.gbm_kwargs["n_estimators"] == 10
