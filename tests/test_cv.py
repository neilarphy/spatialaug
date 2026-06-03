import numpy as np
import pandas as pd
import pytest

from spatialaug.metrics import SpatialBlockCV


@pytest.fixture
def df_uniform_grid():
    lats = np.linspace(55.5, 55.9, 20)
    lons = np.linspace(37.4, 37.8, 20)
    grid_lat, grid_lon = np.meshgrid(lats, lons)
    return pd.DataFrame({"lat": grid_lat.flatten(), "lon": grid_lon.flatten()})


def test_basic_n_splits(df_uniform_grid):
    cv = SpatialBlockCV(n_splits=5, block_size_km=5)
    splits = list(cv.split(df_uniform_grid, "lat", "lon"))
    assert len(splits) == 5


def test_train_test_disjoint(df_uniform_grid):
    cv = SpatialBlockCV(n_splits=5, block_size_km=5)
    for train_idx, test_idx in cv.split(df_uniform_grid, "lat", "lon"):
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_train_test_cover_all(df_uniform_grid):
    cv = SpatialBlockCV(n_splits=5, block_size_km=5)
    n = len(df_uniform_grid)
    for train_idx, test_idx in cv.split(df_uniform_grid, "lat", "lon"):
        assert len(train_idx) + len(test_idx) == n


def test_each_fold_has_both_train_and_test(df_uniform_grid):
    cv = SpatialBlockCV(n_splits=5, block_size_km=5)
    for train_idx, test_idx in cv.split(df_uniform_grid, "lat", "lon"):
        assert len(train_idx) > 0
        assert len(test_idx) > 0


def test_reproducible_with_same_seed(df_uniform_grid):
    cv1 = SpatialBlockCV(n_splits=5, block_size_km=5, random_state=42)
    cv2 = SpatialBlockCV(n_splits=5, block_size_km=5, random_state=42)

    splits1 = [(set(tr), set(te)) for tr, te in cv1.split(df_uniform_grid, "lat", "lon")]
    splits2 = [(set(tr), set(te)) for tr, te in cv2.split(df_uniform_grid, "lat", "lon")]
    assert splits1 == splits2


def test_different_seed_different_split(df_uniform_grid):
    cv1 = SpatialBlockCV(n_splits=5, block_size_km=5, random_state=1)
    cv2 = SpatialBlockCV(n_splits=5, block_size_km=5, random_state=2)

    splits1 = [set(te) for _, te in cv1.split(df_uniform_grid, "lat", "lon")]
    splits2 = [set(te) for _, te in cv2.split(df_uniform_grid, "lat", "lon")]
    assert splits1 != splits2


def test_auto_block_size_works(df_uniform_grid):
    cv = SpatialBlockCV(n_splits=5, random_state=42)
    splits = list(cv.split(df_uniform_grid, "lat", "lon"))
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0


def test_spatial_separation(df_uniform_grid):
    cv = SpatialBlockCV(n_splits=4, block_size_km=10, random_state=42)
    n = len(df_uniform_grid)
    splits = list(cv.split(df_uniform_grid, "lat", "lon"))

    blocked_var = np.mean(
        [
            df_uniform_grid.iloc[test_idx]["lat"].var()
            + df_uniform_grid.iloc[test_idx]["lon"].var()
            for _, test_idx in splits
        ]
    )

    rng = np.random.default_rng(0)
    random_var = np.mean(
        [
            df_uniform_grid.iloc[rng.choice(n, size=len(test_idx), replace=False)]["lat"].var()
            + df_uniform_grid.iloc[rng.choice(n, size=len(test_idx), replace=False)]["lon"].var()
            for _, test_idx in splits
        ]
    )

    assert blocked_var < random_var


def test_invalid_n_splits():
    with pytest.raises(ValueError, match="n_splits must be"):
        SpatialBlockCV(n_splits=1)


def test_invalid_block_size():
    with pytest.raises(ValueError, match="block_size_km must be"):
        SpatialBlockCV(block_size_km=0)


def test_too_few_blocks():
    df = pd.DataFrame({"lat": [55.7, 55.8], "lon": [37.6, 37.7]})
    cv = SpatialBlockCV(n_splits=5, block_size_km=1000)
    with pytest.raises(ValueError, match="Need at least"):
        list(cv.split(df, "lat", "lon"))


def test_missing_columns():
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    cv = SpatialBlockCV()
    with pytest.raises(KeyError, match="not in dataframe"):
        list(cv.split(df, "lat", "lon"))


def test_empty_dataframe():
    cv = SpatialBlockCV()
    with pytest.raises(ValueError, match="Empty"):
        list(cv.split(pd.DataFrame({"lat": [], "lon": []}), "lat", "lon"))


def test_get_n_splits():
    cv = SpatialBlockCV(n_splits=7)
    assert cv.get_n_splits() == 7
