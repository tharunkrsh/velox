"""Tradovate "Performance" CSV (Account Reports > Performance) -> TradeLog.

Columns used: ``symbol, qty, buyPrice, sellPrice, pnl, boughtTimestamp,
soldTimestamp`` (``duration``, fill ids and price-format columns are ignored).

* Each row is a matched buy/sell fill pair. A short trade sells first, so the
  open time is ``min(bought, sold)`` and the close time ``max(bought, sold)``.
* A trade split across partial fills appears as several rows with identical
  symbol and timestamps; those rows are merged into one trade.
* ``pnl`` is accounting-formatted (``$50.00``, ``$(25.00)``) and is GROSS of
  commissions. Pass ``commission_per_unit`` (round turn, per contract) to net it.
* Timestamps are ``MM/DD/YYYY HH:MM:SS`` in the timezone configured for the
  report (usually the user's local zone), so the caller must supply it.
* There is no stop or MAE data in this report.
"""

from __future__ import annotations

import pandas as pd

from velox.domain.trades import TradeLog, TradeLogError
from velox.ingest.generic import Source, _to_number, read_table

REQUIRED = ("symbol", "qty", "pnl", "boughtTimestamp", "soldTimestamp")
FORMAT = "%m/%d/%Y %H:%M:%S"


def read(source: Source, timezone: str, commission_per_unit: float = 0.0) -> TradeLog:
    raw = read_table(source)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise TradeLogError(f"not a Tradovate Performance export; missing columns {missing}")
    raw = raw.dropna(subset=list(REQUIRED))
    try:
        bought = pd.to_datetime(raw["boughtTimestamp"].str.strip(), format=FORMAT)
        sold = pd.to_datetime(raw["soldTimestamp"].str.strip(), format=FORMAT)
        df = pd.DataFrame({
            "symbol": raw["symbol"].str.strip(),
            "bought": bought,
            "sold": sold,
            "units": _to_number(raw["qty"]),
            "pnl": _to_number(raw["pnl"]),
        })
    except (ValueError, TypeError) as e:
        raise TradeLogError(f"could not parse Tradovate report: {e}") from e

    # merge partial fills of the same trade
    df = (df.groupby(["symbol", "bought", "sold"], as_index=False, sort=False)
            .agg(units=("units", "sum"), pnl=("pnl", "sum")))
    df["pnl"] -= commission_per_unit * df["units"]
    try:
        open_t = df[["bought", "sold"]].min(axis=1).dt.tz_localize(
            timezone, ambiguous="raise", nonexistent="raise")
        close_t = df[["bought", "sold"]].max(axis=1).dt.tz_localize(
            timezone, ambiguous="raise", nonexistent="raise")
    except Exception as e:  # unknown zone or DST-ambiguous timestamp
        raise TradeLogError(f"could not localise timestamps to '{timezone}': {e}") from e
    return TradeLog.from_frame(pd.DataFrame({
        "open_time": open_t, "close_time": close_t, "symbol": df["symbol"],
        "units": df["units"], "pnl": df["pnl"],
    }))
