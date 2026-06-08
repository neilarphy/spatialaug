import numpy as np
import pandas as pd
import pytest

from spatialaug.privacy import ValueBasedMask


@pytest.fixture
def fns_like_df():
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "centroid_lat": rng.uniform(55, 56, n),
        "centroid_lon": rng.uniform(37, 38, n),
        "kkt_count": rng.poisson(50, n) + 1, 
        "avg_bill": rng.lognormal(7, 0.5, n),  
    })


def test_k_anonymity_masks_low_kkt(fns_like_df):
    mask = ValueBasedMask(scenario="k_anonymity", percentile_low=5)
    masked, mask_arr = mask.apply(fns_like_df, target="avg_bill")
    p5 = np.percentile(fns_like_df.kkt_count, 5)
    masked_rows = fns_like_df[mask_arr]
    assert (masked_rows.kkt_count <= p5).all()
    # Доля масок около 5%
    assert 0.02 <= mask_arr.mean() <= 0.10


def test_big_business_masks_high_bill(fns_like_df):
    mask = ValueBasedMask(scenario="big_business", percentile_high=95)
    masked, mask_arr = mask.apply(fns_like_df, target="avg_bill")
    p95 = np.percentile(fns_like_df.avg_bill, 95)
    masked_rows = fns_like_df[mask_arr]
    assert (masked_rows.avg_bill >= p95).all()
    assert 0.02 <= mask_arr.mean() <= 0.10


def test_concentration_masks_low_kkt_high_bill(fns_like_df):
    mask = ValueBasedMask(
        scenario="concentration", percentile_low=20, percentile_high=80,
    )
    masked, mask_arr = mask.apply(fns_like_df, target="avg_bill")
    p20_kkt = np.percentile(fns_like_df.kkt_count, 20)
    p80_bill = np.percentile(fns_like_df.avg_bill, 80)
    masked_rows = fns_like_df[mask_arr]
    assert (masked_rows.kkt_count <= p20_kkt).all()
    assert (masked_rows.avg_bill >= p80_bill).all()


def test_target_is_actually_masked(fns_like_df):
    mask = ValueBasedMask(scenario="big_business", percentile_high=95)
    masked, mask_arr = mask.apply(fns_like_df, target="avg_bill")
    assert masked.loc[mask_arr, "avg_bill"].isna().all()
    assert masked.loc[~mask_arr, "avg_bill"].notna().all()


def test_summary_reports_stats(fns_like_df):
    mask = ValueBasedMask(scenario="big_business", percentile_high=95)
    s = mask.summary(fns_like_df)
    assert s["scenario"] == "big_business"
    assert s["n_total"] == 200
    assert s["n_masked"] > 0
    assert 0 < s["ratio"] < 0.15


def test_invalid_scenario_raises():
    with pytest.raises(ValueError, match="scenario"):
        ValueBasedMask(scenario="invalid_scenario") 


def test_invalid_percentiles_raise():
    with pytest.raises(ValueError):
        ValueBasedMask(scenario="big_business", percentile_low=60)
    with pytest.raises(ValueError):
        ValueBasedMask(scenario="big_business", percentile_high=40)


def test_missing_column_raises(fns_like_df):
    mask = ValueBasedMask(scenario="k_anonymity", kkt_col="missing_col")
    with pytest.raises(KeyError):
        mask.apply(fns_like_df, target="avg_bill")


def test_empty_mask_raises():
    df = pd.DataFrame({
        "kkt_count": [1.0, 1.0, 1.0], 
        "avg_bill": [100.0, 100.0, 100.0],
    })
    mask = ValueBasedMask(scenario="big_business", percentile_high=99.9)
    try:
        mask.apply(df, target="avg_bill")
    except ValueError as e:
        assert "пустая" in str(e).lower() or "empty" in str(e).lower()
