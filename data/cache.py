"""
cache.py — Parquet caching layer for VELOX.

Never download the same data twice.
Saves DataFrames as Parquet files locally — fast to read,
compressed, and preserves dtypes perfectly.

Cache file naming convention:
    .cache/data/AAPL_2018-01-01_2023-12-31.parquet
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _cache_path(
    symbol:     str,
    start_date: str,
    end_date:   str,
    cache_dir:  Path,
) -> Path:
    """Build the cache file path for a symbol + date range."""
    filename = f"{symbol}_{start_date}_{end_date}.parquet"
    return cache_dir / filename


def load_from_cache(
    symbol:     str,
    start_date: str,
    end_date:   str,
    cache_dir:  Path,
) -> pd.DataFrame | None:
    """
    Load a DataFrame from cache if it exists.
    Returns None if not cached yet.
    """
    path = _cache_path(symbol, start_date, end_date, cache_dir)

    if not path.exists():
        return None

    try:
        df = pd.read_parquet(path)
        logger.info(f"Cache hit: {path.name}")
        return df
    except Exception as e:
        logger.warning(f"Cache read failed for {symbol}: {e}")
        return None


def save_to_cache(
    df:         pd.DataFrame,
    symbol:     str,
    start_date: str,
    end_date:   str,
    cache_dir:  Path,
) -> None:
    """
    Save a DataFrame to cache as Parquet.
    Creates the cache directory if it doesn't exist.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, start_date, end_date, cache_dir)

    try:
        df.to_parquet(path)
        logger.info(f"Cached: {path.name}")
    except Exception as e:
        logger.warning(f"Cache write failed for {symbol}: {e}")