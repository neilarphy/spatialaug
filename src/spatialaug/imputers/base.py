from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Imputer(ABC):
    """Base class for all imputation methods.

    An imputer fits on observed data (rows where the target is not NaN)
    and then fills missing values in the target column based on spatial
    coordinates and, optionally, additional features.

    Contract:
        - fit(df, lat, lon, target, **kwargs) -> self
        - transform(df) -> pd.DataFrame (a copy with NaN in target filled)
        - fit_transform(df, ...) -> pd.DataFrame
    """

    def __init__(self) -> None:
        self.lat_col: str | None = None
        self.lon_col: str | None = None
        self.target_col: str | None = None
        self.is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        lat: str,
        lon: str,
        target: str,
        **kwargs,
    ) -> Imputer:
        """Fit the imputer on observed rows (where target is not NaN).

        Parameters
        ----------
        df : pd.DataFrame
            Input data with coordinates and target column.
        lat, lon : str
            Column names for latitude and longitude.
        target : str
            Column name to be reconstructed.

        Returns
        -------
        self
        """

        raise NotImplementedError

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN in the target column. Returns a copy of df."""
        
        raise NotImplementedError

    def fit_transform(
        self,
        df: pd.DataFrame,
        lat: str,
        lon: str,
        target: str,
        **kwargs,
    ) -> pd.DataFrame:
        self.fit(df, lat=lat, lon=lon, target=target, **kwargs)
        return self.transform(df)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted via .fit() before .transform()"
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{type(self).__name__}({status})"
