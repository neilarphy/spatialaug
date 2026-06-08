from pathlib import Path

import pytest

from spatialaug.benchmark.datasets import (
    FNS_ALL_AVAILABLE_TARGETS,
    FNS_BUSINESS_TARGETS,
    FNS_DEFAULT_FEATURES,
    FNS_DEFAULT_TARGETS,
    fns_available_cities,
    load_fns_city,
)

DATA_PATH = Path("data/fns/all_cities.parquet")

pytestmark = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="FNS data not available (data/fns/all_cities.parquet)",
)


def test_available_cities_nonempty():
    cities = fns_available_cities()
    assert len(cities) >= 5
    assert "moscow" in cities
    assert "kazan" in cities


def test_load_moscow_default():
    df = load_fns_city("moscow")
    assert len(df) > 0
    assert "centroid_lat" in df.columns
    assert "centroid_lon" in df.columns
    assert "avg_bill" in df.columns
    assert df["avg_bill"].notna().all()


def test_load_with_features():
    df = load_fns_city(
        "moscow",
        target="avg_bill",
        feature_cols=["is_mall", "intensity_bills"],
    )
    assert "is_mall" in df.columns
    assert "intensity_bills" in df.columns
    assert df["is_mall"].dtype.kind in ("i", "u")


def test_unknown_city_raises():
    with pytest.raises(ValueError, match="not found"):
        load_fns_city("atlantis")


def test_unknown_target_raises():
    with pytest.raises(ValueError, match="target"):
        load_fns_city("moscow", target="bitcoin_price")


def test_target_not_in_features():
    df = load_fns_city(
        "moscow",
        target="kkt_count",
        feature_cols=["kkt_count", "intensity_bills"],
    )
    assert list(df.columns).count("kkt_count") == 1


def test_keep_null_target():
    df_dropped = load_fns_city("moscow", target="avg_bill", drop_null_target=True)
    df_kept = load_fns_city("moscow", target="avg_bill", drop_null_target=False)
    assert len(df_kept) >= len(df_dropped)
    if len(df_kept) > len(df_dropped):
        assert df_kept["avg_bill"].isna().sum() > 0


def test_default_targets_loadable():
    for target in FNS_DEFAULT_TARGETS:
        df = load_fns_city("moscow", target=target)
        assert target in df.columns
        assert len(df) > 0


def test_business_targets_loadable():
    assert FNS_BUSINESS_TARGETS == ("avg_bill", "intensity_bills", "cache_pay_pct")
    for target in FNS_BUSINESS_TARGETS:
        df = load_fns_city("moscow", target=target)
        assert target in df.columns
        assert len(df) > 0


def test_all_available_targets_loadable():
    for target in FNS_ALL_AVAILABLE_TARGETS:
        df = load_fns_city("moscow", target=target)
        assert target in df.columns


def test_default_features_constant():
    assert "is_mall" in FNS_DEFAULT_FEATURES
    assert "intensity_bills" in FNS_DEFAULT_FEATURES
