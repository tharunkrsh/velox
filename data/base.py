"""
[base.py](http://base.py) — Abstract base class for all data handlers in VELOX.

Defines the contract every data source must fulfil.

The engine only ever talks to this interface — never to

yfinance, Alpaca, or any specific source directly.

This means swapping historical → live data requires

zero changes to the engine. Just a different handler.
"""

from abc import ABC, abstractmethod
import pandas as pd


class DataHandler(ABC):
    """
    Abstract base for all data handlers.

    Subclasses must implement:
    update_bars()  — advance the clock one bar
    has_next()     — is there more data?
    get_latest()   — retrieve recent bars for a symbol
    """

    # Injected by the engine at startup
    events = None

    @abstractmethod
    def update_bars(self) -> None:
        """
        Push the next bar of data for all symbols into
        latest_bars, then fire a MarketEvent.

        This is called once per iteration of the outer
        engine loop — it IS the clock tick.
        """
        raise NotImplementedError

    @abstractmethod
    def has_next(self) -> bool:
        """
        Returns True if there is at least one more bar
        of data available across all symbols.
        """
        raise NotImplementedError

    @abstractmethod
    def get_latest(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """
        Return the last N bars for a symbol as a DataFrame.

        Columns: open, high, low, close, volume
        Index:   datetime

        Strategies call this to read price history.
        Crucially — they can ONLY see bars up to NOW.
        Calling this can never return future data.
        """
        raise NotImplementedError