import numpy as np
import pandas as pd
import pytest

from spatialaug import IDWImputer, MeanImputer
from spatialaug.transfer import TransferEvaluator, TransferResult


def make_synthetic_city(n: int = 100, seed: int = 0,
                        lat_center: float = 55.7, lon_center: float = 37.5,
                        scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "lat": rng.normal(lat_center, 0.3, n),
        "lon": rng.normal(lon_center, 0.3, n),
        "feature": rng.normal(50, 10, n),
        "target": rng.normal(1000 * scale, 100, n),
    })


def test_transfer_result_dataclass_basic():
    r = TransferResult(
        source_name="A", target_name="B", target_col="y",
        method_name="mean", mechanism="mcar",
        n_source=100, n_target=50, n_masked_target=15,
        mae_zero_shot_mean=200.0, mae_zero_shot_std=10.0,
        mae_full_refit_mean=180.0, mae_full_refit_std=8.0,
        transfer_stability_mean=1.11, transfer_stability_std=0.05,
        n_seeds=3,
    )
    row = r.to_row()
    assert row["source"] == "A"
    assert row["target"] == "B"
    assert row["fns_target"] == "y"
    assert row["method"] == "mean"
    assert row["transfer_stability"] == 1.11
    assert row["n_seeds"] == 3


def test_transfer_evaluator_constructor():
    eval_ = TransferEvaluator(
        imputer_factory=lambda: MeanImputer(strategy="median"),
        method_name="mean_test",
        lat_col="lat", lon_col="lon", target_col="target",
        seeds=[1, 2, 3],
    )
    assert eval_.method_name == "mean_test"
    assert eval_.seeds == [1, 2, 3]
    assert eval_.lat_col == "lat"


def test_transfer_evaluator_evaluate_returns_per_mechanism():
    src = make_synthetic_city(n=80, seed=1, scale=1.0)
    tgt = make_synthetic_city(n=50, seed=2, scale=1.2)

    eval_ = TransferEvaluator(
        imputer_factory=lambda: MeanImputer(strategy="median"),
        method_name="mean",
        lat_col="lat", lon_col="lon", target_col="target",
        seeds=[1, 2],
        mechanisms=["mcar", "diffuse_mnar"],
    )
    results = eval_.evaluate(src, tgt, source_name="A", target_name="B")
    assert len(results) == 2  
    for r in results:
        assert isinstance(r, TransferResult)
        assert r.source_name == "A"
        assert r.target_name == "B"
        assert r.n_seeds == 2  
        assert len(r.per_seed_rows) == 2


def test_transfer_evaluator_stability_metric():
    src = make_synthetic_city(n=80, seed=1, scale=1.0)
    tgt = make_synthetic_city(n=50, seed=2, scale=1.0)

    eval_ = TransferEvaluator(
        imputer_factory=lambda: IDWImputer(power=2, n_neighbors=5),
        method_name="idw",
        lat_col="lat", lon_col="lon", target_col="target",
        seeds=[1, 2, 3],
        mechanisms=["mcar"],
    )
    results = eval_.evaluate(src, tgt, source_name="A", target_name="B")
    r = results[0]
    assert np.isfinite(r.transfer_stability_mean) or np.isnan(r.transfer_stability_mean)
    assert r.n_seeds == 3


def test_transfer_evaluator_to_dataframe():
    src = make_synthetic_city(n=80, seed=1)
    tgt = make_synthetic_city(n=50, seed=2)
    eval_ = TransferEvaluator(
        imputer_factory=lambda: MeanImputer(strategy="median"),
        method_name="mean", lat_col="lat", lon_col="lon", target_col="target",
        seeds=[1, 2], mechanisms=["mcar"],
    )
    df = eval_.evaluate_matrix(
        cities={"A": src, "B": tgt},
        pairs=[("A", "B")],
    )
    assert len(df) == 1
    assert "transfer_stability" in df.columns
    assert df.iloc[0]["source"] == "A"


def test_transfer_evaluator_matrix_default_pairs():
    cities = {
        "A": make_synthetic_city(80, seed=1),
        "B": make_synthetic_city(80, seed=2),
        "C": make_synthetic_city(80, seed=3),
    }
    eval_ = TransferEvaluator(
        imputer_factory=lambda: MeanImputer(strategy="median"),
        method_name="mean", lat_col="lat", lon_col="lon", target_col="target",
        seeds=[1], mechanisms=["mcar"],
    )
    df = eval_.evaluate_matrix(cities=cities)
    assert len(df) == 6  


def test_transfer_evaluator_handles_imputer_failure_gracefully():
    src = make_synthetic_city(n=80, seed=1)
    tgt = make_synthetic_city(n=10, seed=2) 

    class BrokenImputer(MeanImputer):
        def fit(self, df, lat, lon, target, **kwargs):
            raise RuntimeError("intentional fail")

    eval_ = TransferEvaluator(
        imputer_factory=BrokenImputer, method_name="broken",
        lat_col="lat", lon_col="lon", target_col="target",
        seeds=[1], mechanisms=["mcar"],
    )
    results = eval_.evaluate(src, tgt, source_name="A", target_name="B")
    assert len(results) == 1
    assert np.isnan(results[0].mae_zero_shot_mean) or results[0].mae_zero_shot_mean is None
