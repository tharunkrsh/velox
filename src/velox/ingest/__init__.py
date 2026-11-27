"""Trade-log ingestion adapters. Each returns a canonical TradeLog."""

from __future__ import annotations

from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from velox.domain.trades import TradeLog
from velox.ingest import mt5, tradovate
from velox.ingest.generic import ColumnMapping, Source, read_csv, read_table

Format = Literal["generic", "mt5", "tradovate"]
FORMATS: tuple[str, ...] = ("generic", "mt5", "tradovate")

# formats whose P&L column is gross of commissions; every other format is already net
GROSS_PNL_FORMATS: tuple[str, ...] = ("tradovate",)

# canonical CSV: open_time, close_time, pnl [, symbol, units, risk, mae]
CANONICAL = ColumnMapping(open_time="open_time", close_time="close_time", pnl="pnl")


def load_trades(
    source: Source,
    fmt: Format = "generic",
    timezone: str | None = None,
    mapping: ColumnMapping | None = None,
    commission_per_unit: float = 0.0,
) -> TradeLog:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format '{fmt}' (choose from {', '.join(FORMATS)})")
    if commission_per_unit < 0:
        raise ValueError("commission per unit must be >= 0")
    if commission_per_unit and fmt not in GROSS_PNL_FORMATS:
        raise ValueError(
            f"a commission per unit only applies to {', '.join(GROSS_PNL_FORMATS)} exports, "
            f"which report gross P&L; '{fmt}' P&L is already net of commission")
    if timezone is not None:
        _check_timezone(timezone)
    if fmt in ("mt5", "tradovate") and timezone is None:
        raise ValueError(f"{fmt} exports have no UTC offset; pass the export's timezone (--tz)")
    if fmt == "mt5":
        return mt5.read(source, timezone)
    if fmt == "tradovate":
        return tradovate.read(source, timezone, commission_per_unit)
    m = mapping or _infer_canonical(source)
    if timezone is not None:
        m = m.model_copy(update={"timezone": timezone})
    return read_csv(source, m)


def _check_timezone(name: str) -> None:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ValueError(f"unknown timezone '{name}'; use an IANA name such as "
                         "Europe/Prague or America/Chicago") from e


def _infer_canonical(source: Source) -> ColumnMapping:
    header = list(read_table(source).columns)
    optional = {c: c for c in ("symbol", "units", "risk", "mae") if c in header}
    return CANONICAL.model_copy(update=optional)
