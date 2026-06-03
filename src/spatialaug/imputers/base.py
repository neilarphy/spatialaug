from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Imputer(ABC):
    """Базовый класс для методов импутации.

    импутер обучается на наблюдаемых данных и затем восстанавливает пропуски
    в целевой колонке на основе пространственных координат и опциональной группы.

    Контракт:
        - fit(df, lat, lon, target, **kwargs) -> self
        - transform(df) -> pd.DataFrame (копия с заполненными NaN в target)
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
        """Обучить импутер на наблюдаемых строках (где target не NaN).

        Parameters
        ----------
        df : pd.DataFrame
            Входные данные с координатами и целевой колонкой.
        lat, lon : str
            Имена колонок широты и долготы.
        target : str
            Имя колонки, которую надо восстановить.

        Returns
        -------
        self
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Восстановить NaN в target-колонке. Возвращает копию df."""
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
