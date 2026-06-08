"""TabPFN foundation-model imputer wrapper.

TabPFN (https://github.com/PriorLabs/TabPFN) is a transformer-based
foundation model for tabular regression (Hollmann et al., ICLR 2023
and NeurIPS 2024). The v2 weights (Prior-Labs/TabPFN-v2-reg on
Hugging Face) are released under Apache 2.0 — see the PriorLabs
repository for the full license. Local inference does NOT require
any PriorLabs API authentication (that is only used by their cloud
app).

TabPFN is an in-context learner: predictions for each query depend
on the provided training context rather than on parameters learned
across datasets. The published sweet spot is n_train <= 1000 rows
and up to ~100 features; latency and accuracy degrade beyond that.
Inference cost scales roughly linearly with the n_estimators
ensemble size.

Install:
    pip install tabpfn  # auto-downloads weights from HF or GCS mirror

Usage:
    from spatialaug import TabPFNImputer
    imp = TabPFNImputer(feature_cols=["kkt_count", "intensity_bills"])
    imp.fit(df, lat="centroid_lat", lon="centroid_lon", target="avg_bill")
    df_filled = imp.transform(df)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from spatialaug.imputers.base import Imputer

TABPFN_GCS_REGRESSOR_URL = (
    "https://storage.googleapis.com/tabpfn-v2-model-files/"
    "05152025/tabpfn-v2-regressor.ckpt"
)
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/tabpfn")


def _ensure_model_downloaded(model_path: str | None = None) -> str:
    """Download TabPFN v2 regressor weights via GCS mirror if not cached.

    Returns the path to the downloaded file.
    """
    if model_path is None:
        os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)
        model_path = os.path.join(DEFAULT_CACHE_DIR, "tabpfn-v2-regressor.ckpt")
    if not os.path.exists(model_path):
        import urllib.request
        urllib.request.urlretrieve(TABPFN_GCS_REGRESSOR_URL, model_path)
    return model_path


class TabPFNImputer(Imputer):
    """Foundation-model imputer backed by TabPFN v2.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ICL-pass ensemble members. More gives better
        accuracy but scales latency linearly.
    feature_cols : list[str] or None
        Features in addition to lat/lon. If None, coordinates only.
    device : "cuda" | "cpu" | "auto", default="auto"
        Execution device. "cuda" is ~10x faster than "cpu".
    model_path : str or None
        Path to the downloaded weights. If None, weights are
        auto-downloaded to ~/.cache/tabpfn/.
    ignore_pretraining_limits : bool, default=True
        Allow fitting on n_train > 10 000 rows (TabPFN's sweet spot
        is <= 1000). Set to True for general use.

    Notes
    -----
    Dependency: pip install tabpfn. We do not vendor it inside
    spatialaug — the user picks their own runtime (GPU/CPU, model
    version). If tabpfn is not installed, fit raises a clear
    ImportError on the first call.

    Examples
    --------
    >>> from spatialaug import TabPFNImputer
    >>> imp = TabPFNImputer(feature_cols=["kkt_count", "intensity_bills"])
    >>> imp.fit(df, lat="centroid_lat", lon="centroid_lon", target="avg_bill")
    >>> df_filled = imp.transform(df)
    """

    def __init__(
        self,
        n_estimators: int = 8,
        feature_cols: list[str] | tuple[str, ...] | None = None,
        device: str = "auto",
        model_path: str | None = None,
        ignore_pretraining_limits: bool = True,
    ) -> None:
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.feature_cols = list(feature_cols) if feature_cols else None
        self.device = device
        self.model_path = model_path
        self.ignore_pretraining_limits = bool(ignore_pretraining_limits)
        self._model = None
        self._actual_device: str | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def _build_X(self, df: pd.DataFrame) -> np.ndarray:
        cols = [self.lat_col, self.lon_col]
        if self.feature_cols:
            cols = cols + list(self.feature_cols)
        return df[cols].astype(float).to_numpy()

    def fit(
        self, df: pd.DataFrame, lat: str, lon: str, target: str, **kwargs,
    ) -> "TabPFNImputer":
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as exc: 
            raise ImportError(
                "TabPFNImputer requires `pip install tabpfn`. "
                "See https://github.com/PriorLabs/TabPFN"
            ) from exc

        self.lat_col, self.lon_col, self.target_col = lat, lon, target
        observed = df.dropna(subset=[target])
        X = self._build_X(observed)
        y = observed[target].astype(float).to_numpy()

        model_path = _ensure_model_downloaded(self.model_path)
        self._actual_device = self._resolve_device(self.device)

        model_kwargs: dict = {
            "n_estimators": self.n_estimators,
            "device": self._actual_device,
            "model_path": model_path,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
        }

        self._model = TabPFNRegressor(**model_kwargs)
        self._model.fit(X, y)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()

        result = df.copy()
        mask = result[self.target_col].isna()
        
        if mask.sum() == 0:
            return result
        cols = [self.lat_col, self.lon_col]
        
        if self.feature_cols:
            cols = cols + list(self.feature_cols)
        
        X = result.loc[mask, cols].astype(float).to_numpy()
        pred = self._model.predict(X)
        result.loc[mask, self.target_col] = pred
        
        return result
