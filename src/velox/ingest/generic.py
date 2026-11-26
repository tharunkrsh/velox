"""Generic column-mapped CSV ingestion -> TradeLog.

Every broker/journal export is reduced to a :class:`ColumnMapping`. The source
timezone is mandatory unless the timestamps carry an explicit UTC offset:
broker server time (e.g. MT5) is rarely UTC, and getting it wrong shifts trades
across daily-loss boundaries.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from velox.domain.trades import TradeLog, TradeLogError


class ColumnMapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    open_time: str
    close_time: str
    pnl: str
    symbol: str | None = None
    units: str | None = None
    risk: str | None = None
    mae: str | None = None
    extra_pnl: list[str] = Field(default_factory=list)  # e.g. commission, swap: added to pnl
    timezone: str | None = None        # IANA zone of naive timestamps, e.g. "Europe/Prague"
    datetime_format: str | None = None  # strftime format; None = let pandas infer
    dayfirst: bool = False


def _to_number(s: pd.Series) -> pd.Series:
    """Parse money-ish text: '$1,250.00', '1 250.00', '$(25.00)' (accounting negative)."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = s.astype(str).str.replace(r"[\s\u00a0$\u20ac\u00a3,]", "", regex=True)
    cleaned = cleaned.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    return pd.to_numeric(cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan}),
                         errors="raise")


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a delimited export regardless of encoding (UTF-8/UTF-16 with BOM, as MT5
    desktop writes) and delimiter (comma, tab or semicolon). All cells as text."""
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    header = text.splitlines()[0] if text else ""
    sep = "\t" if "\t" in header else (";" if header.count(";") > header.count(",") else ",")
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, skipinitialspace=True,
                     keep_default_na=False, na_values=[""])
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_utc(s: pd.Series, m: ColumnMapping, name: str) -> pd.Series:
    ts = pd.to_datetime(s, format=m.datetime_format, dayfirst=m.dayfirst, errors="raise")
    if ts.dt.tz is None:
        if m.timezone is None:
            raise TradeLogError(
                f"column '{name}' has no UTC offset; pass the source timezone "
                "(e.g. --tz Europe/Prague). Broker server time is usually not UTC."
            )
        ts = ts.dt.tz_localize(m.timezone, ambiguous="raise", nonexistent="raise")
    return ts.dt.tz_convert("UTC")


def frame_to_tradelog(raw: pd.DataFrame, m: ColumnMapping) -> TradeLog:
    cols = {"open_time": m.open_time, "close_time": m.close_time, "pnl": m.pnl,
            "symbol": m.symbol, "units": m.units, "risk": m.risk, "mae": m.mae}
    wanted = [c for c in [*cols.values(), *m.extra_pnl] if c is not None]
    missing = [c for c in wanted if c not in raw.columns]
    if missing:
        raise TradeLogError(f"columns not found in file: {missing}; available: {list(raw.columns)}")

    df = pd.DataFrame(index=raw.index)
    try:
        df["open_time"] = _to_utc(raw[m.open_time], m, m.open_time)
        df["close_time"] = _to_utc(raw[m.close_time], m, m.close_time)
        df["pnl"] = _to_number(raw[m.pnl])
        for extra in m.extra_pnl:
            df["pnl"] += _to_number(raw[extra]).fillna(0.0)
        for canon in ("units", "risk", "mae"):
            if cols[canon] is not None:
                df[canon] = _to_number(raw[cols[canon]])
    except (ValueError, TypeError) as e:
        if isinstance(e, TradeLogError):
            raise
        raise TradeLogError(f"could not parse trade log: {e}") from e
    if m.symbol is not None:
        df["symbol"] = raw[m.symbol].astype(str).str.strip()
    return TradeLog.from_frame(df)


def read_csv(path: str | Path, m: ColumnMapping) -> TradeLog:
    return frame_to_tradelog(read_table(path), m)
