import numpy as np

from spatialaug.metrics import by_target_quartile, compute_error_suite


def test_perfect_predictions():
    y_true = np.array([100.0, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    y_pred = y_true.copy()
    s = compute_error_suite(y_true, y_pred)
    assert s["mae"] == 0
    assert s["rmse"] == 0
    assert s["mape"] == 0
    assert s["r2"] == 1.0
    assert s["bias"] == 0


def test_constant_offset_pred():
    y_true = np.array([100.0, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    y_pred = y_true + 50  
    s = compute_error_suite(y_true, y_pred)
    assert s["mae"] == 50
    assert s["bias"] == 50 
    assert s["median_error"] == 50


def test_mape_skips_zeros():
    y_true = np.array([0.0, 100, 200])
    y_pred = np.array([1.0, 110, 220])
    s = compute_error_suite(y_true, y_pred)
    expected = np.mean([10/100, 20/200]) * 100
    assert abs(s["mape"] - expected) < 0.01


def test_r2_correct():
    rng = np.random.RandomState(0)
    y_true = rng.normal(100, 20, 100)
    y_pred = y_true + rng.normal(0, 5, 100) 
    s = compute_error_suite(y_true, y_pred)
    assert s["r2"] > 0.9


def test_heteroscedasticity_grows_with_target():
    rng = np.random.RandomState(0)
    y_true = np.linspace(10, 1000, 100)
    y_pred = y_true + rng.normal(0, y_true * 0.1)
    s = compute_error_suite(y_true, y_pred)
    assert s["heteroscedasticity_ratio"] > 2.0


def test_quartile_breakdown():
    rng = np.random.RandomState(0)
    y_true = np.linspace(10, 1000, 100)
    y_pred = y_true + rng.normal(0, 10, 100)
    q = by_target_quartile(y_true, y_pred)
    assert sorted(q.keys()) == ["Q1", "Q2", "Q3", "Q4"]
    for label in q:
        assert q[label]["n"] > 0
        assert q[label]["mae"] >= 0


def test_invalid_input_filtered():
    y_true = np.array([1.0, 2, np.nan, 4])
    y_pred = np.array([1.1, 2.0, 3.0, np.inf])
    s = compute_error_suite(y_true, y_pred)
    assert s["n_used"] == 2  
