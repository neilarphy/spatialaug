"""Extended error metrics (beyond MAE / RMSE).

Designed for business-oriented interpretation of imputation quality:

- MAPE — Mean Absolute Percentage Error. Error as a percent of the
  true value. Often more meaningful for financial targets than MAE.
- R-squared — variance explained.
- Bias — mean(prediction - truth). Detects systematic over/under
  estimation.
- Heteroscedasticity ratio — std(error | top y-quartile) divided by
  std(error | bottom y-quartile). Captures whether error grows with
  target magnitude.
- Median absolute error — robust to outliers.
"""

from __future__ import annotations

import numpy as np


def compute_error_suite(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    skip_invalid: bool = True,
) -> dict[str, float]:
    """Compute the full error suite in one pass.

    Parameters
    ----------
    y_true, y_pred : (N,) arrays
        Ground truth and predicted values.
    skip_invalid : bool, default=True
        If True, drop pairs where y_true or y_pred is NaN/Inf. MAPE
        additionally skips pairs with y_true near zero.

    Returns
    -------
    dict
        Keys: mae, rmse, mape, r2, bias, median_error,
        heteroscedasticity_ratio, n_used.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")

    if skip_invalid:
        valid = (
            np.isfinite(y_true) & np.isfinite(y_pred)
        )
        y_true = y_true[valid]
        y_pred = y_pred[valid]

    n = len(y_true)
    if n == 0:
        return {k: float("nan") for k in [
            "mae", "rmse", "mape", "r2", "bias", "median_error",
            "heteroscedasticity_ratio",
        ]} | {"n_used": 0}

    error = y_pred - y_true
    abs_error = np.abs(error)

    mae = float(abs_error.mean())
    rmse = float(np.sqrt((error ** 2).mean()))
    bias = float(error.mean()) 
    median_error = float(np.median(abs_error))

    nonzero = np.abs(y_true) > 1e-9
    if nonzero.sum() > 0:
        mape = float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100)
    else:
        mape = float("nan")

    ss_res = float(np.sum(error ** 2))
    y_mean = y_true.mean()
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    if n >= 8:
        q1, q3 = np.quantile(y_true, [0.25, 0.75])
        bottom = abs_error[y_true <= q1]
        top = abs_error[y_true >= q3]
        if bottom.std() > 1e-9:
            hetero = float(top.std() / bottom.std())
        else:
            hetero = float("nan")
    else:
        hetero = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "bias": bias,
        "median_error": median_error,
        "heteroscedasticity_ratio": hetero,
        "n_used": int(n),
    }


def by_target_quartile(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Per-quartile error breakdown — where does the model err most?

    Splits y_true into quartiles Q1..Q4 and reports MAE, RMSE, bias
    and the y-range within each bin.

    Returns
    -------
    dict
        Maps quartile label ("Q1".."Q4") to a dict with keys n, mae,
        rmse, bias, y_true_range.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) < 4:
        return {}

    q = np.quantile(y_true, [0.25, 0.5, 0.75])
    bins = np.digitize(y_true, q)
    result = {}
    for i, label in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        mask = bins == i
        if mask.sum() == 0:
            continue
        err = y_pred[mask] - y_true[mask]
        result[label] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(err.mean()),
            "y_true_range": (float(y_true[mask].min()), float(y_true[mask].max())),
        }
    
    return result
