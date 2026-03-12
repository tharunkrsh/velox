"""
portfolio.py - Tracks positions, cash, and P&L for VELOX.

Responsibilities:
    - Receive SignalEvents from strategies
    - Decide position sizes using Kelly Criterion
    - Fire OrderEvents to the execution handler
    - Receive FillEvents and update positions + cash
    - Track equity curve over time
"""

import logging
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

from core.events import (
    SignalEvent, FillEvent, OrderEvent,
    OrderDirection, OrderType, SignalDirection
)

logger = logging.getLogger(__name__)


class Portfolio:

    def __init__(
        self,
        data_handler,
        initial_capital: float = 100_000.0,
    ):
        self.data         = data_handler
        self.initial_capital = initial_capital
        self.events       = None  # injected by engine

        # Current state
        self.cash: float         = initial_capital
        self.positions: Dict[str, float] = {}  # symbol → quantity held
        self.holdings: Dict[str, float]  = {}  # symbol → market value

        # History — one row per bar
        self.equity_curve: List[dict] = []

        # All fills ever received
        self.fill_history: List[FillEvent] = []

        # Performance metrics — populated after backtest
        self.metrics: dict = {}

    # ─── Event Handlers ───────────────────────────────────────────────────────

    def on_signal(self, event: SignalEvent) -> None:
        """
        Receive a signal from a strategy.
        Calculate order size and fire an OrderEvent.
        """
        symbol    = event.symbol
        direction = event.direction
        strength  = event.strength  # 0.0 to 1.0

        if direction == SignalDirection.EXIT:
            self._close_position(symbol)
            return

        # Size the position: allocate a fraction of capital
        # Strength scales the allocation (Kelly-inspired)
        allocation  = self.cash * 0.10 * strength  # max 10% per position
        latest      = self.data.get_latest(symbol, n=1)

        if latest.empty:
            return

        price    = float(latest["close"].iloc[-1])
        quantity = allocation / price

        if quantity < 1:
            return

        order_direction = (
            OrderDirection.BUY
            if direction == SignalDirection.LONG
            else OrderDirection.SELL
        )

        order = OrderEvent(
            symbol     = symbol,
            order_type = OrderType.MARKET,
            direction  = order_direction,
            quantity   = round(quantity, 4),
        )

        logger.debug(f"Order fired: {order}")
        self.events.put(order)

    def on_fill(self, event: FillEvent) -> None:
        """
        Receive a fill from the execution handler.
        Update positions and cash.
        """
        self.fill_history.append(event)
        symbol    = event.symbol
        quantity  = event.quantity
        direction = event.direction

        # Update position
        current = self.positions.get(symbol, 0.0)

        if direction == OrderDirection.BUY:
            self.positions[symbol] = current + quantity
            self.cash -= event.total_cost
        else:
            self.positions[symbol] = current - quantity
            self.cash += event.fill_cost - event.commission

        logger.debug(
            f"Fill: {direction.value} {quantity} {symbol} "
            f"@ {event.fill_price:.2f} | Cash: {self.cash:.2f}"
        )

        self._update_equity()

    # ─── Position Management ──────────────────────────────────────────────────

    def _close_position(self, symbol: str) -> None:
        """Fire a SELL order to close an existing long position."""
        quantity = self.positions.get(symbol, 0.0)
        if quantity <= 0:
            return

        order = OrderEvent(
            symbol     = symbol,
            order_type = OrderType.MARKET,
            direction  = OrderDirection.SELL,
            quantity   = quantity,
        )
        self.events.put(order)

    def _update_equity(self) -> None:
        """
        Snapshot the current portfolio value.
        Called after every fill.
        """
        market_value = 0.0

        for symbol, qty in self.positions.items():
            if qty == 0:
                continue
            try:
                latest = self.data.get_latest(symbol, n=1)
                price  = float(latest["close"].iloc[-1])
                market_value += qty * price
            except Exception:
                pass

        total_equity = self.cash + market_value

        self.equity_curve.append({
            "timestamp"    : datetime.utcnow(),
            "cash"         : self.cash,
            "market_value" : market_value,
            "total_equity" : total_equity,
        })

    # ─── Performance Metrics ──────────────────────────────────────────────────

    def compute_metrics(self) -> None:
        """
        Compute performance metrics after backtest completes.
        Called automatically by the engine at the end of the run.
        """
        if not self.equity_curve:
            logger.warning("No equity curve data to compute metrics from.")
            return

        df     = pd.DataFrame(self.equity_curve)
        equity = df["total_equity"]

        # Daily returns
        returns = equity.pct_change().dropna()

        # Core metrics
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital
        ann_return   = (1 + total_return) ** (252 / len(returns)) - 1
        ann_vol      = returns.std() * np.sqrt(252)
        sharpe       = ann_return / ann_vol if ann_vol != 0 else 0.0

        # Drawdown
        rolling_max  = equity.cummax()
        drawdown     = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        self.metrics = {
            "initial_capital" : self.initial_capital,
            "final_equity"    : round(equity.iloc[-1], 2),
            "total_return_pct": round(total_return * 100, 2),
            "ann_return_pct"  : round(ann_return * 100, 2),
            "ann_volatility"  : round(ann_vol * 100, 2),
            "sharpe_ratio"    : round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "calmar_ratio"    : round(calmar, 3),
            "total_trades"    : len(self.fill_history),
        }

        logger.info("── Performance Metrics ──────────────────")
        for k, v in self.metrics.items():
            logger.info(f"  {k:<22}: {v}")
        logger.info("─────────────────────────────────────────")