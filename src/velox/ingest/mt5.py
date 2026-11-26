"""MetaTrader 5 "Positions" history export -> TradeLog.

Layout (Toolbox > History > Positions; desktop export or copy/paste):

    Time, Position, Symbol, Type, Volume, Price, S / L, T / P, Time, Price,
    Commission, Swap, Profit

* The duplicated Time/Price headers are open then close; pandas reads the second
  ones as ``Time.1`` / ``Price.1``.
* The desktop terminal writes UTF-16 with a BOM and tab separators; copies and
  web exports are UTF-8 with commas. Both are accepted.
* Timestamps are broker *server* time (``2024.01.15 10:30:00``), so the caller
  must supply the server timezone.
* Profit may carry a space thousands separator (``1 250.00``); S/L and T/P may be
  blank. Risk is not derived from S/L because that needs per-symbol contract
  sizes, and MT5 does not export MAE.
* Rows without a close time or symbol (balance/summary lines) are dropped.
"""

from __future__ import annotations

from pathlib import Path

from velox.domain.trades import TradeLog
from velox.ingest.generic import ColumnMapping, frame_to_tradelog, read_table


def mapping(timezone: str) -> ColumnMapping:
    return ColumnMapping(
        open_time="Time",
        close_time="Time.1",
        symbol="Symbol",
        units="Volume",
        pnl="Profit",
        extra_pnl=["Commission", "Swap"],
        timezone=timezone,
        datetime_format="%Y.%m.%d %H:%M:%S",
    )


def read(path: str | Path, timezone: str) -> TradeLog:
    raw = read_table(path)
    if {"Time.1", "Symbol"} <= set(raw.columns):
        raw = raw.dropna(subset=["Time.1", "Symbol"]).reset_index(drop=True)
    return frame_to_tradelog(raw, mapping(timezone))
