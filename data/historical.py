"""
historical.py - yfinance-backed historical data handler.

Downloads OHLCV data for a list of symbols, caches it
locally as Parquet, then replays it bar by bar.

The "time machine" pattern:
    all_bars     = the full dataset (never shown to strategies)
    latest_bars  = only what the engine has seen so far
                   (grows by one bar per update_bars() call)
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

from core.events import MarketEvent
from data.base import DataHandler
from data.cache import load_from_cache, save_to_cache

logger = logging.getLogger(__name__)


class HistoricalDataHandler(DataHandler):

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d",
        cache_dir: str = ".cache/data",
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.cache_dir = Path(cache_dir)

        self.all_bars: dict = {}
        self.latest_bars: dict = defaultdict(list)

        self._index: pd.DatetimeIndex = None
        self._bar_pos: int = 0

        self._load_all()

    def _load_all(self) -> None:
        frames = {}

        for symbol in self.symbols:
            df = self._fetch(symbol)
            if df is not None and not df.empty:
                frames[symbol] = df
                logger.info(f"Loaded {len(df)} bars for {symbol}")
            else:
                logger.warning(f"No data for {symbol}, skipping.")

        if not frames:
            raise ValueError("No data loaded. Check symbols and date range.")

        common_index = frames[list(frames.keys())[0]].index
        for df in frames.values():
            common_index = common_index.intersection(df.index)

        for symbol, df in frames.items():
            self.all_bars[symbol] = df.loc[common_index]

        self._index = common_index
        logger.info(
            f"Data aligned. {len(self._index)} common bars across "
            f"{len(self.all_bars)} symbols."
        )

    def _fetch(self, symbol: str) -> pd.DataFrame:
        cached = load_from_cache(symbol, self.start_date, self.end_date, self.cache_dir)
        if cached is not None:
            logger.info(f"{symbol}: loaded from cache")
            return cached

        logger.info(f"{symbol}: downloading from yfinance...")
        try:
            raw = yf.download(
                symbol,
                start=self.start_date,
                end=self.end_date,
                interval=self.interval,
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                return None

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0].lower() for c in raw.columns]
            else:
                raw.columns = [c.lower() for c in raw.columns]
            raw.index.name = "datetime"

            save_to_cache(raw, symbol, self.start_date, self.end_date, self.cache_dir)
            return raw

        except Exception as e:
            logger.error(f"Failed to download {symbol}: {e}")
            return None

    def has_next(self) -> bool:
        return self._bar_pos < len(self._index)

    def update_bars(self) -> None:
        current_dt = self._index[self._bar_pos]

        for symbol in self.all_bars:
            bar = self.all_bars[symbol].loc[current_dt]
            self.latest_bars[symbol].append((current_dt, bar))

        self._bar_pos += 1
        self.events.put(MarketEvent(timestamp=current_dt))

    def get_latest(self, symbol: str, n: int = 1) -> pd.DataFrame:
        if symbol not in self.latest_bars:
            raise KeyError(f"Symbol '{symbol}' not found in data handler.")

        bars = self.latest_bars[symbol][-n:]

        if not bars:
            return pd.DataFrame()

        index = [b[0] for b in bars]
        data = [b[1] for b in bars]

        return pd.DataFrame(data, index=index)