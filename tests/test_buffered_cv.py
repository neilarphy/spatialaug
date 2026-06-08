import numpy as np
import pandas as pd
import pytest

from spatialaug.metrics import SpatialBlockCV


def make_grid_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "lat": rng.uniform(55, 56, n),
        "lon": rng.uniform(37, 38, n),
        "y": rng.normal(0, 1, n),
    })


def test_no_buffer_default_behavior_unchanged():
    df = make_grid_df(n=200)
    cv = SpatialBlockCV(n_splits=5, block_size_km=20, buffer_km=0,
                        random_state=42)
    splits_no_buf = list(cv.split(df, lat_col="lat", lon_col="lon"))

    cv_default = SpatialBlockCV(n_splits=5, block_size_km=20, random_state=42)
    splits_default = list(cv_default.split(df, lat_col="lat", lon_col="lon"))

    for (tr1, te1), (tr2, te2) in zip(splits_no_buf, splits_default, strict=True):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)


def test_buffer_reduces_train_size():
    df = make_grid_df(n=200)
    cv_no = SpatialBlockCV(n_splits=5, block_size_km=20, buffer_km=0)
    cv_buf = SpatialBlockCV(n_splits=5, block_size_km=20, buffer_km=10)

    no_buf_train_sizes = [len(tr) for tr, te in cv_no.split(df, "lat", "lon")]
    buf_train_sizes = [len(tr) for tr, te in cv_buf.split(df, "lat", "lon")]

    for nb, b in zip(no_buf_train_sizes, buf_train_sizes, strict=True):
        assert b <= nb


def test_buffer_no_overlap_with_test():
    df = make_grid_df(n=300)
    cv = SpatialBlockCV(n_splits=5, block_size_km=15, buffer_km=5)
    for tr_idx, te_idx in cv.split(df, "lat", "lon"):
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        tr_lats = df.iloc[tr_idx]["lat"].to_numpy()
        tr_lons = df.iloc[tr_idx]["lon"].to_numpy()
        te_lats = df.iloc[te_idx]["lat"].to_numpy()
        te_lons = df.iloc[te_idx]["lon"].to_numpy()

        import math
        center_lat = float(df["lat"].mean())
        lat_diff_km = (tr_lats[:, None] - te_lats[None, :]) * 111.0
        lon_diff_km = (
            (tr_lons[:, None] - te_lons[None, :])
            * 111.0 * math.cos(math.radians(center_lat))
        )
        dist_km = np.sqrt(lat_diff_km ** 2 + lon_diff_km ** 2)
        min_dist = dist_km.min(axis=1)
        assert (min_dist > 5).all()


def test_negative_buffer_raises():
    with pytest.raises(ValueError, match="buffer_km"):
        SpatialBlockCV(buffer_km=-1.0)


def test_buffer_test_indices_unchanged():
    df = make_grid_df(n=200)
    cv_no = SpatialBlockCV(n_splits=5, block_size_km=20, buffer_km=0,
                            random_state=42)
    cv_buf = SpatialBlockCV(n_splits=5, block_size_km=20, buffer_km=10,
                             random_state=42)

    for (_, te_no), (_, te_buf) in zip(
        cv_no.split(df, "lat", "lon"),
        cv_buf.split(df, "lat", "lon"), strict=True,
    ):
        np.testing.assert_array_equal(te_no, te_buf)
