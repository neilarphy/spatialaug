"""FNS receipts — public hex-grid retail-POS dataset.

This is a domain-specific adapter for the Russian FNS (Federal Tax
Service) open hex-grid dataset that we used as the primary benchmark
corpus. Each row is an H3 hexagon (resolution 8, ~0.74 km^2) in one
of 11 Russian cities, carrying aggregates over the cash-registers
inside the cell.

Available target columns
------------------------
- avg_bill — mean receipt size (RUB).
- median_bill — median receipt size.
- trunc_avg_bill — mean receipt size with outliers trimmed.
- kkt_count — number of registers in the hex.
- cache_bill_pct — % of receipts paid in cash.
- cache_pay_pct — % of payments made in cash.
- intensity_bills — decile-rank intensity of receipts.
- intensity_revenue — decile-rank intensity of revenue.

Features available for feature_cols
-----------------------------------
- is_mall, is_rare, is_ecommerce — categorical flags (bool, cast to int).
- intensity_bills, intensity_revenue — decile-rank indices.
- kkt_count — register count.

Natural privacy MNAR
--------------------
About 23 % of hexes (~808 of 3531) carry NaN in avg_bill / median_bill —
the tax authority suppresses values when a hex contains very few
businesses, for privacy. drop_null_target=True drops them by default;
set it to False to keep this natural MNAR for dedicated experiments.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/fns/all_cities.parquet")

GEO_COLS = ("centroid_lat", "centroid_lon")

ALL_AVAILABLE_TARGETS = (
    # Financial (continuous, RUB)
    "avg_bill",           
    "median_bill",        
    "trunc_avg_bill",     
    # Behavioral (bounded 0..100 %)
    "cache_pay_pct",       
    "cache_bill_pct",     
    # Transactional (ordinal 1..10)
    "intensity_bills",     
    "intensity_revenue",  
    # Infrastructure (count)
    "kkt_count",           
    "receipt_count",       
)

BUSINESS_TARGETS = (
    "avg_bill",        
    "intensity_bills",  
    "cache_pay_pct",   
)

DEFAULT_TARGETS = ("avg_bill", "kkt_count")

DEFAULT_FEATURES = (
    "is_mall",
    "is_rare",
    "is_ecommerce",
    "kkt_count",
    "intensity_bills",
    "intensity_revenue",
)


def load_fns_city(
    city: str,
    target: str = "avg_bill",
    *,
    drop_null_target: bool = True,
    feature_cols: list[str] | tuple[str, ...] | None = None,
    data_path: Path = DATA_PATH,
) -> pd.DataFrame:
    """Load hex-level data for a single FNS city.

    Parameters
    ----------
    city : str
        One of: moscow, spb, kazan, krasnodar, novosibirsk, voronezh,
        kaliningrad, ekaterinburg, chelyabinsk, yakutsk, magadan.
    target : str, default="avg_bill"
        Target column to predict.
    drop_null_target : bool, default=True
        If True, drop hexes whose target is NaN. Set to False to
        keep the naturally NaN rows (privacy MNAR) for dedicated
        experiments.
    feature_cols : list or tuple of str, optional
        Extra feature columns to keep. Defaults to DEFAULT_FEATURES
        with target itself removed if it would otherwise appear.
    data_path : Path
        Path to the merged-cities parquet file.

    Returns
    -------
    pd.DataFrame
        Columns: centroid_lat, centroid_lon, <target>, plus the
        requested features.
    """

    if not data_path.exists():
        raise FileNotFoundError(f"FNS data file not found at {data_path}")
    
    df = pd.read_parquet(data_path)
    sub = df[df["city"] == city].copy()
    
    if sub.empty:
        available = sorted(df["city"].unique())
        raise ValueError(
            f"city {city!r} not found in dataset. Available: {available}"
        )
    
    if target not in sub.columns:
        raise ValueError(
            f"target {target!r} not found in dataset. "
            f"Available columns: {sorted(df.columns)}"
        )
    
    if drop_null_target:
        sub = sub.dropna(subset=[target])

    
    keep = list(GEO_COLS) + [target]
    
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURES
    
    for col in feature_cols:
        if col == target:
            continue
        if col in sub.columns and col not in keep:
            keep.append(col)

    sub = sub[keep].copy()

    for col in sub.columns:
        if sub[col].dtype == bool:
            sub[col] = sub[col].astype(int)
    sub = sub.dropna(subset=list(GEO_COLS))

    return sub.reset_index(drop=True)


def available_cities(data_path: Path = DATA_PATH) -> list[str]:
    """Return cities in the dataset sorted by number of hexes (desc)."""
    
    if not data_path.exists():
        return []
    
    df = pd.read_parquet(data_path, columns=["city"])
    counts = df["city"].value_counts()
    
    return list(counts.index)
