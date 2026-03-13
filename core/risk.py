"""
risk.py - Risk management layer for VELOX.

Sits between the portfolio and execution handler.
Acts as a gatekeeper — filters or modifies orders
that would violate risk limits before they get filled.

Risk rules enforced:
    - Max position size (% of portfolio)
    - Max number of open positions
    - Max drawdown circuit breaker (halts all trading)
"""

import logging
from typing import Dict

from core.events import OrderEvent, OrderDirection

logger = logging.getLogger(__name__)


class RiskManager:

    def __init__(
        self,
        portfolio,
        max_position_pct:   float = 0.20,   # max 20% of portfolio in one stock
        max_open_positions: int   = 10,      # max 10 simultaneous positions
        max_drawdown_pct:   float = 0.20,    # halt trading if down 20%
    ):
        self.portfolio           = portfolio
        self.max_position_pct    = max_position_pct
        self.max_open_positions  = max_open_positions
        self.max_drawdown_pct    = max_drawdown_pct
        self.trading_halted      = False

    def validate(self, order: OrderEvent) -> bool:
        """
        Returns True if the order passes all risk checks.
        Returns False if the order should be rejected.
        """

        if self.trading_halted:
            logger.warning("Trading halted — max drawdown breached.")
            return False

        if not self._check_drawdown():
            self.trading_halted = True
            logger.warning("Max drawdown breached — halting all trading.")
            return False

        if order.direction == OrderDirection.BUY:
            if not self._check_position_count():
                logger.warning(
                    f"Max open positions ({self.max_open_positions}) reached. "
                    f"Order for {order.symbol} rejected."
                )
                return False

            if not self._check_position_size(order):
                logger.warning(
                    f"Position size limit breached for {order.symbol}. "
                    f"Order rejected."
                )
                return False

        return True

    def _check_drawdown(self) -> bool:
        """Returns False if current drawdown exceeds the limit."""
        curve = self.portfolio.equity_curve
        if len(curve) < 2:
            return True

        peak   = max(e["total_equity"] for e in curve)
        latest = curve[-1]["total_equity"]

        if peak == 0:
            return True

        drawdown = (peak - latest) / peak
        return drawdown < self.max_drawdown_pct

    def _check_position_count(self) -> bool:
        """Returns False if too many positions are already open."""
        open_positions = sum(
            1 for qty in self.portfolio.positions.values()
            if qty > 0
        )
        return open_positions < self.max_open_positions

    def _check_position_size(self, order: OrderEvent) -> bool:
        """Returns False if the order would exceed max position size."""
        equity = self.portfolio.equity_curve
        if not equity:
            return True

        total_equity = equity[-1]["total_equity"]
        if total_equity == 0:
            return True

        latest = self.portfolio.data.get_latest(order.symbol, n=1)
        if latest.empty:
            return True

        price         = float(latest["close"].iloc[-1])
        order_value   = order.quantity * price
        position_pct  = order_value / total_equity

        return position_pct <= self.max_position_pct