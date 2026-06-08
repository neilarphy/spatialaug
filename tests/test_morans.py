import numpy as np

from spatialaug.metrics import morans_i, morans_preservation_ratio


def test_constant_values_returns_zero():
    coords = np.random.RandomState(0).uniform(0, 1, (100, 2))
    values = np.ones(100) * 42.0
    i, _ = morans_i(coords, values)
    assert i == 0.0


def test_random_scatter_near_zero():
    rng = np.random.RandomState(42)
    coords = rng.uniform(0, 1, (500, 2))
    values = rng.normal(0, 1, 500)
    i, _ = morans_i(coords, values)
    assert abs(i) < 0.1


def test_clustered_values_positive_high():
    rng = np.random.RandomState(1)
    coords = rng.uniform(0, 1, (300, 2))
    values = coords[:, 0]  
    i, _ = morans_i(coords, values)
    assert i > 0.5


def test_anti_correlated_negative():
    xs, ys = np.meshgrid(np.arange(10), np.arange(10))
    coords = np.column_stack([xs.ravel(), ys.ravel()])
    values = ((xs + ys) % 2 == 0).astype(float).ravel() * 2 - 1
    i, _ = morans_i(coords, values, k_neighbors=4)
    assert i < 0


def test_permutation_p_value_works():
    rng = np.random.RandomState(0)
    coords = rng.uniform(0, 1, (200, 2))
    values = coords[:, 0]
    i, p = morans_i(coords, values, n_permutations=100, random_state=0)
    assert i > 0.5
    assert p < 0.1


def test_preservation_ratio_perfect():
    rng = np.random.RandomState(7)
    coords = rng.uniform(0, 1, (100, 2))
    values = coords[:, 0] + 0.1 * rng.normal(size=100)
    result = morans_preservation_ratio(coords, values, values.copy())
    assert abs(result["ratio"] - 1.0) < 1e-9
    assert abs(result["delta"]) < 1e-9


def test_preservation_ratio_noise_destroys_structure():
    rng = np.random.RandomState(3)
    coords = rng.uniform(0, 1, (200, 2))
    original = coords[:, 0]
    noisy = original + rng.normal(0, original.std() * 5, size=200)
    result = morans_preservation_ratio(coords, original, noisy)
    assert result["ratio"] < 0.5
