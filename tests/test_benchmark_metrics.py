import math

import numpy as np
import pytest

from spatialaug.benchmark import degradation_slope, delta_f1, stability_score


def test_stability_score_perfect():
    assert stability_score([1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_stability_score_high_variance():
    s = stability_score([1.0, 10.0])
    assert s < 0.5


def test_stability_score_empty():
    assert math.isnan(stability_score([]))


def test_stability_score_zero_mean():
    assert math.isnan(stability_score([-1.0, 1.0]))


def test_degradation_slope_negative_for_decreasing():
    slope = degradation_slope([0.1, 0.3, 0.5, 0.7, 1.0], [10.0, 8.0, 6.0, 4.0, 2.0])
    assert slope < 0


def test_degradation_slope_positive_for_increasing():
    slope = degradation_slope([0.1, 0.5, 1.0], [1.0, 2.0, 3.0])
    assert slope > 0


def test_degradation_slope_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        degradation_slope([0.1, 0.5], [1.0])


def test_degradation_slope_too_few_points():
    with pytest.raises(ValueError, match="at least 2"):
        degradation_slope([0.5], [1.0])


def test_delta_f1_positive():
    assert delta_f1(0.85, 0.80) == pytest.approx(0.05)


def test_delta_f1_negative():
    assert delta_f1(0.70, 0.80) == pytest.approx(-0.10)


def test_delta_f1_zero():
    assert delta_f1(0.80, 0.80) == pytest.approx(0.0)
