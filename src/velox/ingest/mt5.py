"""MetaTrader 5 "Positions" history export -> TradeLog.

Expected layout (MT5 Toolbox > History > Positions, exported/copied as CSV):

    Time, Position, Symbol, Type, Volume, Price, S / L, T / P, Time, Price,
    Commission, Swap, Profit

The duplicated Time/Price headers are open then close; pandas reads the second
ones as ``Time.1`` / ``Price.1``. Timestamps are broker *server* time
(``2024.01.15 10:30:00``), so the caller must supply the server timezone.

Risk is not derived from S/L because that needs per-symbol contract sizes.

Status: built from the documented column layout, not yet validated against a
real export. Send a sample to confirm before relying on it.
"""

from __future__ import annotations

from pathlib import Path

from velox.domain.trades import TradeLog
from velox.ingest.generic import ColumnMapping, read_csv


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
    return read_csv(path, mapping(timezone))
