"""Value-based privacy missingness mechanisms.

Emulates the privacy-driven redaction rules that statistical
agencies, payment regulators and tax authorities apply to retail or
POS data world-wide: instead of a random or radius-based mask, rows
are masked when their feature values trigger one of three
regulatory scenarios that recur across jurisdictions.

1. k_anonymity — count column below the lower percentile: too few
   units per cell, aggregates are not anonymous and revenue-side
   fields are redacted.

2. big_business — value column above the upper percentile: an
   anomalously large average indicates a large entity whose
   commercial information must not be exposed. Counterpart to the
   primary-suppression rules used by US Census Bureau, Eurostat,
   etc.

3. concentration — count below the lower percentile AND value above
   the upper percentile: revenue concentrates on a handful of units,
   so averaging would expose individual turnover.

Default column names (kkt_count, avg_bill) reflect a typical
retail-POS schema (count of terminals and mean receipt size), but
kkt_col and bill_col are fully configurable — pass any analogous
pair of count and value columns to apply the same scenarios on a
different domain.

Validation: when the source dataset already contains naturally NaN
rows induced by the regulator, comparing the distributions of those
natural-NaN rows against the rows produced by this masker is a
direct check.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Scenario = Literal["k_anonymity", "big_business", "concentration"]


_SCENARIO_RULES: dict[str, tuple[tuple[str, str], ...]] = {
    "k_anonymity":   (("kkt_col", "low"),),
    "big_business":  (("bill_col", "high"),),
    "concentration": (("kkt_col", "low"), ("bill_col", "high")),
}


class ValueBasedMask:
    """Privacy-induced masking driven by value-based thresholds.

    Parameters
    ----------
    scenario : {"k_anonymity", "big_business", "concentration"}
        Which regulatory scenario to emulate. See module docstring.
    percentile_low : float, default=5
        Lower percentile threshold (applied to kkt_col under
        k_anonymity and concentration).
    percentile_high : float, default=95
        Upper percentile threshold (applied to bill_col under
        big_business and concentration).
    kkt_col : str, default="kkt_count"
        Column carrying the cash-register count (used by k_anonymity
        and concentration).
    bill_col : str, default="avg_bill"
        Column carrying the average receipt size (used by
        big_business and concentration).
    random_state : int, default=42
        Unused — the value-based mask is deterministic. Kept only for
        API parity with MissingnessSimulator.

    Examples
    --------
    >>> mask = ValueBasedMask(scenario="big_business", percentile_high=95)
    >>> df_masked, idx = mask.apply(df, target="avg_bill")
    >>> # df_masked["avg_bill"] is NaN where original avg_bill > p95
    """

    def __init__(
        self,
        scenario: Scenario,
        percentile_low: float = 5.0,
        percentile_high: float = 95.0,
        kkt_col: str = "kkt_count",
        bill_col: str = "avg_bill",
        random_state: int = 42,
    ) -> None:
        
        if scenario not in _SCENARIO_RULES:
            raise ValueError(
                f"scenario must be one of {tuple(_SCENARIO_RULES)}, got {scenario!r}"
            )
        
        if not 0 < percentile_low < 50:
            raise ValueError(f"percentile_low must be in (0, 50), got {percentile_low}")
        
        if not 50 < percentile_high < 100:
            raise ValueError(f"percentile_high must be in (50, 100), got {percentile_high}")
        
        self.scenario = scenario
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.kkt_col = kkt_col
        self.bill_col = bill_col
        self.random_state = random_state

    def _compute_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Return a boolean array where True marks rows to be masked.

        Iterates over the atomic rules of the chosen scenario (see
        _SCENARIO_RULES). Each rule contributes one threshold check
        on a single column plus a NaN-exclusion, AND-combined into
        the final mask.
        """
        mask = np.ones(len(df), dtype=bool)
        for col_attr, direction in _SCENARIO_RULES[self.scenario]:
            col = getattr(self, col_attr)
            if col not in df.columns:
                raise KeyError(
                    f"{self.scenario} scenario needs column {col!r}"
                )
            values = df[col].to_numpy(dtype=float)
            pct = self.percentile_low if direction == "low" else self.percentile_high
            thresh = np.nanpercentile(values, pct)
            if direction == "low":
                mask &= values <= thresh
            else:
                mask &= values >= thresh
            mask &= ~np.isnan(values)
        
        return mask

    def apply(
        self, df: pd.DataFrame, target: str,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Apply the value-based mask to ``target``.

        Parameters
        ----------
        df : pd.DataFrame
            Input data. Must contain the columns required by the
            chosen scenario (kkt_col and/or bill_col).
        target : str
            Column whose values are set to NaN where the mask is True.

        Returns
        -------
        masked_df : pd.DataFrame
            A copy of df with the target column set to NaN on masked
            rows.
        mask : np.ndarray
            Boolean array of length len(df). True marks the rows that
            were masked.
        """
        mask = self._compute_mask(df)

        if mask.sum() == 0:
            raise ValueError(
                f"Empty mask: no rows satisfy the {self.scenario} criterion. "
                "Percentile thresholds may be too extreme, or the target "
                "is already all-NaN."
            )
        
        masked = df.copy()
        masked.loc[mask, target] = np.nan
        
        return masked, mask

    def summary(self, df: pd.DataFrame) -> dict:
        """Report how many rows fall under the mask for the given df."""
        mask = self._compute_mask(df)

        n = len(df)
        
        return {
            "scenario": self.scenario,
            "n_total": n,
            "n_masked": int(mask.sum()),
            "ratio": float(mask.sum() / n) if n > 0 else 0.0,
            "percentile_low": self.percentile_low,
            "percentile_high": self.percentile_high,
        }
