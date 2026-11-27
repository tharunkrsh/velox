"""Data-quality checks that are surfaced to the user next to every result."""

from __future__ import annotations

import datetime as dt

import numpy as np
from pydantic import BaseModel

from velox.domain.calendar import SessionCalendar
from velox.domain.sizing import IntratradeModel, resolve_intratrade
from velox.domain.trades import TradeLog

MIN_DAYS = 30
MIN_TRADES = 100


class DataQualityReport(BaseModel):
    n_trades: int
    n_trading_days: int
    first_trade: dt.datetime
    last_trade: dt.datetime
    overlapping_trades: int
    has_risk: bool
    has_mae: bool
    intratrade_model: IntratradeModel
    warnings: list[str]


def assess(
    log: TradeLog,
    calendar: SessionCalendar,
    intratrade: IntratradeModel = IntratradeModel.AUTO,
) -> DataQualityReport:
    warnings: list[str] = []
    model = resolve_intratrade(log, intratrade)
    n_days = len(np.unique(calendar.trading_day(log.close_time)))

    # trade i overlaps if it opened before the latest close among earlier-closing trades
    opens = log.open_time.asi8
    closes = log.close_time.asi8
    prev_max_close = np.maximum.accumulate(np.concatenate([[np.iinfo(np.int64).min], closes[:-1]]))
    overlaps = int(np.sum(opens < prev_max_close))

    if n_days < MIN_DAYS or len(log) < MIN_TRADES:
        warnings.append(
            f"Small sample ({len(log)} trades over {n_days} trading days; recommended >= "
            f"{MIN_TRADES} trades and >= {MIN_DAYS} days). Probabilities will be noisy and "
            "may overstate your edge."
        )
    if overlaps:
        warnings.append(
            f"{overlaps} trades overlap an earlier open trade. VELOX v1 treats trades as "
            "sequential, which understates floating drawdown when positions are held concurrently."
        )
    if model is IntratradeModel.REALIZED:
        warnings.append(
            "No MAE data: intra-trade drawdown is assumed equal to the realised loss. Daily-loss "
            "and trailing breaches on equity are therefore UNDER-estimated (optimistic). Add an "
            "MAE column or use the stop_bound model for a pessimistic bound."
        )
    if not log.has_risk:
        warnings.append("No per-trade risk column: risk-per-trade sizing advice is in units only.")

    return DataQualityReport(
        n_trades=len(log),
        n_trading_days=n_days,
        first_trade=log.open_time.min().to_pydatetime(),
        last_trade=log.close_time.max().to_pydatetime(),
        overlapping_trades=overlaps,
        has_risk=log.has_risk,
        has_mae=log.has_mae,
        intratrade_model=model,
        warnings=warnings,
    )
