import numpy as np
import pandas as pd
import pytest

from spatialaug import TabPFNImputer
from spatialaug.imputers.base import Imputer


def test_tabpfn_is_imputer_subclass():
    assert issubclass(TabPFNImputer, Imputer)


def test_tabpfn_default_params():
    imp = TabPFNImputer()
    assert imp.n_estimators == 8
    assert imp.feature_cols is None
    assert imp.device == "auto"
    assert imp.model_path is None
    assert imp.ignore_pretraining_limits is True


def test_tabpfn_with_feature_cols():
    imp = TabPFNImputer(feature_cols=["kkt_count", "is_mall"])
    assert imp.feature_cols == ["kkt_count", "is_mall"]


def test_tabpfn_n_estimators_coerced_to_int():
    imp = TabPFNImputer(n_estimators=4)
    assert imp.n_estimators == 4
    assert isinstance(imp.n_estimators, int)


def test_tabpfn_resolve_device_auto():
    device = TabPFNImputer._resolve_device("auto")
    assert device in ("cuda", "cpu")


def test_tabpfn_resolve_device_explicit():
    assert TabPFNImputer._resolve_device("cpu") == "cpu"
    assert TabPFNImputer._resolve_device("cuda") == "cuda"


def test_tabpfn_build_X_with_features():
    imp = TabPFNImputer(feature_cols=["kkt_count"])
    imp.lat_col, imp.lon_col, imp.target_col = "lat", "lon", "y"
    df = pd.DataFrame({
        "lat": [55.7, 55.8], "lon": [37.5, 37.6],
        "kkt_count": [100, 200], "y": [1000, 2000],
    })
    X = imp._build_X(df)
    assert X.shape == (2, 3)  
    assert X[0, 2] == 100


def test_tabpfn_build_X_no_features():
    imp = TabPFNImputer()
    imp.lat_col, imp.lon_col = "lat", "lon"
    df = pd.DataFrame({"lat": [55.7], "lon": [37.5], "y": [100.0]})
    X = imp._build_X(df)
    assert X.shape == (1, 2)


def test_tabpfn_transform_without_fit_raises():
    imp = TabPFNImputer()
    df = pd.DataFrame({"lat": [55.7], "lon": [37.5], "y": [np.nan]})
    with pytest.raises(RuntimeError):
        imp.transform(df)


def test_tabpfn_in_main_imports():
    import spatialaug
    assert "TabPFNImputer" in spatialaug.__all__
