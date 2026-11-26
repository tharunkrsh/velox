"""Canonical trade log: the single normalised input to every simulation.

A ``TradeLog`` is columnar (NumPy arrays, one row per closed trade) and sorted
by close time. Timestamps are timezone-aware UTC; mapping trades to firm
trading days is the job of :mod:`velox.domain.calendar`, not of ingestion.

Money columns are in account currency:
    pnl   net realised P&L of the trade (commission/swap already included)
    risk  planned loss at the stop, as a positive amount (NaN if unknown)
    mae   max adverse excursion while open, as a positive amount (NaN if unknown)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class TradeLogError(ValueError):
    """Raised when a trade log cannot be normalised into a valid TradeLog."""


@dataclass(frozen=True)
class TradeLog:
    open_time: pd.DatetimeIndex   # tz-aware UTC
    close_time: pd.DatetimeIndex  # tz-aware UTC
    symbol: np.ndarray            # object[str]
    units: np.ndarray             # float64, > 0 (lots, contracts, shares)
    pnl: np.ndarray               # float64
    risk: np.ndarray              # float64, NaN where unknown
    mae: np.ndarray               # float64, NaN where unknown

    def __post_init__(self) -> None:
        n = len(self.pnl)
        cols = {
            "open_time": self.open_time, "close_time": self.close_time, "symbol": self.symbol,
            "units": self.units, "risk": self.risk, "mae": self.mae,
        }
        for name, col in cols.items():
            if len(col) != n:
                raise TradeLogError(f"column '{name}' has {len(col)} rows, expected {n}")
        for name, idx in (("open_time", self.open_time), ("close_time", self.close_time)):
            if idx.tz is None or str(idx.tz) != "UTC":
                raise TradeLogError(f"'{name}' must be tz-aware UTC")
        if n == 0:
            raise TradeLogError("trade log is empty")
        if not np.all(np.isfinite(self.pnl)):
            raise TradeLogError("pnl contains NaN/inf")
        if not np.all(self.units > 0):
            raise TradeLogError("units must be > 0 for every trade")
        if np.any(self.close_time < self.open_time):
            raise TradeLogError("close_time earlier than open_time for some trades")
        if np.any(np.diff(self.close_time.asi8) < 0):
            raise TradeLogError("trades must be sorted by close_time")
        for name, col in (("risk", self.risk), ("mae", self.mae)):
            if np.any(col[~np.isnan(col)] < 0):
                raise TradeLogError(f"'{name}' must be a positive amount")

    def __len__(self) -> int:
        return len(self.pnl)

    @property
    def has_risk(self) -> bool:
        return bool(np.all(~np.isnan(self.risk)))

    @property
    def has_mae(self) -> bool:
        return bool(np.all(~np.isnan(self.mae)))

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> TradeLog:
        """Build from a frame with canonical columns; sorts by close_time.

        Required: open_time, close_time, pnl. Optional: symbol, units, risk, mae.
        Datetime columns must already be tz-aware.
        """
        missing = {"open_time", "close_time", "pnl"} - set(df.columns)
        if missing:
            raise TradeLogError(f"missing required columns: {sorted(missing)}")
        df = df.sort_values("close_time", kind="stable").reset_index(drop=True)
        n = len(df)

        def _num(name: str, default: float) -> np.ndarray:
            if name not in df.columns:
                return np.full(n, default, dtype=float)
            return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)

        def _utc(name: str) -> pd.DatetimeIndex:
            idx = pd.DatetimeIndex(df[name])
            if idx.tz is None:
                raise TradeLogError(f"'{name}' is timezone-naive; declare the source timezone")
            return idx.tz_convert("UTC")

        symbol = (df["symbol"].astype(str).to_numpy(dtype=object) if "symbol" in df.columns
                  else np.full(n, "", dtype=object))
        return cls(
            open_time=_utc("open_time"),
            close_time=_utc("close_time"),
            symbol=symbol,
            units=_num("units", 1.0),
            pnl=_num("pnl", np.nan),
            risk=_num("risk", np.nan),
            mae=np.abs(_num("mae", np.nan)),
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "open_time": self.open_time, "close_time": self.close_time, "symbol": self.symbol,
            "units": self.units, "pnl": self.pnl, "risk": self.risk, "mae": self.mae,
        })
