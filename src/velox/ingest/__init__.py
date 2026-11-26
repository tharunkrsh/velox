"""Trade-log ingestion adapters. Each returns a canonical TradeLog."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from velox.domain.trades import TradeLog
from velox.ingest import mt5
from velox.ingest.generic import ColumnMapping, read_csv

Format = Literal["generic", "mt5"]

# canonical CSV: open_time, close_time, pnl [, symbol, units, risk, mae]
CANONICAL = ColumnMapping(open_time="open_time", close_time="close_time", pnl="pnl")


def load_trades(
    path: str | Path,
    fmt: Format = "generic",
    timezone: str | None = None,
    mapping: ColumnMapping | None = None,
) -> TradeLog:
    if fmt == "mt5":
        if timezone is None:
            raise ValueError("MT5 exports are in broker server time; pass the server timezone")
        return mt5.read(path, timezone)
    m = mapping or _infer_canonical(path)
    if timezone is not None:
        m = m.model_copy(update={"timezone": timezone})
    return read_csv(path, m)


def _infer_canonical(path: str | Path) -> ColumnMapping:
    import pandas as pd

    header = [c.strip() for c in pd.read_csv(path, nrows=0).columns]
    optional = {c: c for c in ("symbol", "units", "risk", "mae") if c in header}
    return CANONICAL.model_copy(update=optional)
