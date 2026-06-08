from __future__ import annotations

import pandas as pd

from spatialaug.imputers.base import Imputer


class MeanImputer(Imputer):
    """Fill missing values with a group-wise mean or median (e.g. by region).

    Uses mean by default. If a group column is provided, computes the
    aggregate within each group. For groups not seen during fit, falls
    back to the global aggregate.

    Parameters
    ----------
    strategy : {"mean", "median"}, default="mean"
        Aggregation function applied to the target.

    Examples
    --------
    >>> imp = MeanImputer(strategy="median")
    >>> df_filled = imp.fit_transform(
    ...     df, lat="geo_lat", lon="geo_lon", target="price", group="region"
    ... )
    """

    def __init__(self, strategy: str = "mean") -> None:

        super().__init__()
        
        if strategy not in ("mean", "median"):
            raise ValueError(f"strategy must be 'mean' or 'median', got {strategy!r}")
        
        self.strategy = strategy
        self.group_col: str | None = None
        self.group_values_: dict | None = None
        self.global_value_: float | None = None

    def fit(
        self,
        df: pd.DataFrame,
        lat: str,
        lon: str,
        target: str,
        group: str | None = None,
        **kwargs,
    ) -> MeanImputer:
        self.lat_col = lat
        self.lon_col = lon
        self.target_col = target
        self.group_col = group

        observed = df[df[target].notna()]
        if observed.empty:
            raise ValueError(f"No observed (non-NaN) rows in target column {target!r}; cannot fit")

        agg = self.strategy
        self.global_value_ = float(getattr(observed[target], agg)())

        if group is not None:
            if group not in df.columns:
                raise KeyError(f"group column {group!r} not in dataframe")
            grouped = observed.groupby(group)[target].agg(agg)
            self.group_values_ = grouped.to_dict()
        else:
            self.group_values_ = None

        self.is_fitted = True
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        self._check_fitted()
        result = df.copy()
        mask = result[self.target_col].isna()
        
        if not mask.any():
            return result

        if self.group_col is not None and self.group_values_:
            group_fill = result.loc[mask, self.group_col].map(self.group_values_)
            group_fill = group_fill.fillna(self.global_value_)
            result.loc[mask, self.target_col] = group_fill.values
        else:
            result.loc[mask, self.target_col] = self.global_value_

        return result
