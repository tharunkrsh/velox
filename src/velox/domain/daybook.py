"""Trades grouped into firm trading days: the unit the simulator resamples.

Stored CSR-style (``offsets[d]:offsets[d+1]`` are the trades of day d) with P&L
and adverse excursion carried *per unit*, so any sizing multiplier can be
applied later without touching the grouping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from velox.domain.calendar import SessionCalendar
from velox.domain.sizing import IntratradeModel, SizingModel, adverse_amount, resolve_intratrade
from velox.domain.trades import TradeLog


@dataclass(frozen=True)
class DayBook:
    days: np.ndarray             # datetime64[D], ascending, one per day with >=1 trade
    offsets: np.ndarray          # int64, len(days) + 1
    units: np.ndarray            # per trade
    pnl_per_unit: np.ndarray     # per trade
    adverse_per_unit: np.ndarray # per trade, >= 0
    intratrade: IntratradeModel  # resolved model used for adverse_per_unit

    @property
    def n_days(self) -> int:
        return len(self.days)

    @property
    def n_trades(self) -> int:
        return len(self.units)

    def day_slice(self, d: int) -> slice:
        return slice(int(self.offsets[d]), int(self.offsets[d + 1]))

    def scaled(self, k: float, sizing: SizingModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-trade (pnl, adverse, units) at multiplier k."""
        units_k = sizing.scale_units(self.units, k)
        return self.pnl_per_unit * units_k, self.adverse_per_unit * units_k, units_k

    def haircut(self, h: float) -> DayBook:
        """Stress test: remove a fraction ``h`` of the average edge from every trade.

        Each trade loses ``h * mean_trade_pnl`` (scaled by its own size), so the mean
        trade P&L at 1.0x becomes ``(1 - h) * mean``. Adverse excursion is kept at
        least as deep as the new realised loss.
        """
        if not 0 <= h <= 1:
            raise ValueError("haircut must be in [0, 1]")
        mu = float(np.mean(self.pnl_per_unit * self.units))
        pnl_pu = self.pnl_per_unit - h * max(mu, 0.0) / self.units
        adverse_pu = np.maximum(self.adverse_per_unit, -pnl_pu)
        return DayBook(self.days, self.offsets, self.units, pnl_pu, adverse_pu, self.intratrade)

    def daily_pnl(self, k: float = 1.0, sizing: SizingModel | None = None) -> np.ndarray:
        pnl, _, _ = self.scaled(k, sizing or SizingModel())
        return np.add.reduceat(pnl, self.offsets[:-1]) if self.n_trades else np.zeros(0)


def build_daybook(
    log: TradeLog,
    calendar: SessionCalendar,
    intratrade: IntratradeModel = IntratradeModel.AUTO,
) -> DayBook:
    model = resolve_intratrade(log, intratrade)
    day = calendar.trading_day(log.close_time)
    # log is sorted by close_time, so day labels are already non-decreasing
    days, starts = np.unique(day, return_index=True)
    offsets = np.append(starts, len(day)).astype(np.int64)
    return DayBook(
        days=days,
        offsets=offsets,
        units=log.units.copy(),
        pnl_per_unit=log.pnl / log.units,
        adverse_per_unit=adverse_amount(log, model) / log.units,
        intratrade=model,
    )
