"""
portfolio.py - Tracks positions, cash, and P&L for VELOX.

Supports both long and short positions.

Long position:  positive quantity, costs cash to open, returns cash to close
Short position: negative quantity, returns cash to open, costs cash to close
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

    def __init__(self, data_handler, initial_capital: float = 100_000.0):
        self.data            = data_handler
        self.initial_capital = initial_capital
        self.events          = None

        self.cash: float                 = initial_capital
        self.positions: Dict[str, float] = {}
        self.avg_price: Dict[str, float] = {}

        self.equity_curve: List[dict]  = []
        self.fill_history: List[FillEvent] = []
        self.metrics: dict             = {}

    def on_signal(self, event: SignalEvent) -> None:
        symbol    = event.symbol
        direction = event.direction
        strength  = event.strength

        if direction == SignalDirection.EXIT:
            self._close_position(symbol)
            return

        latest = self.data.get_latest(symbol, n=1)
        if latest.empty:
            return

        price      = float(latest["close"].iloc[-1])
        allocation = self.cash * 0.10 * strength
        quantity   = allocation / price

        if quantity < 1:
            return

        if direction == SignalDirection.LONG:
            if self.positions.get(symbol, 0) > 0:
                return
            order_direction = OrderDirection.BUY

        elif direction == SignalDirection.SHORT:
            if self.positions.get(symbol, 0) < 0:
                return
            order_direction = OrderDirection.SELL

        else:
            return

        order = OrderEvent(
            symbol     = symbol,
            order_type = OrderType.MARKET,
            direction  = order_direction,
            quantity   = round(quantity, 4),
        )
        self.events.put(order)

    def on_fill(self, event: FillEvent) -> None:
        self.fill_history.append(event)
        symbol      = event.symbol
        quantity    = event.quantity
        direction   = event.direction
        price       = event.fill_price
        current_qty = self.positions.get(symbol, 0.0)

        if direction == OrderDirection.BUY:
            if current_qty < 0:
                # Covering a short — just pay the cost to buy back
                self.cash -= event.total_cost
                new_qty    = current_qty + quantity
            else:
                self.cash -= event.total_cost
                new_qty    = current_qty + quantity

            if new_qty > 0:
                old_cost = current_qty * self.avg_price.get(symbol, price)
                new_cost = quantity * price
                self.avg_price[symbol] = (old_cost + new_cost) / new_qty
            else:
                self.avg_price[symbol] = 0.0

        else:  # SELL
            if current_qty > 0:
                self.cash += event.fill_cost - event.commission
                new_qty    = current_qty - quantity
            else:
                self.cash += event.fill_cost - event.commission
                new_qty    = current_qty - quantity
                self.avg_price[symbol] = price

        self.positions[symbol] = new_qty

        logger.debug(
            f"Fill: {direction.value} {quantity:.2f} {symbol} "
            f"@ {price:.2f} | Position: {new_qty:.2f} | Cash: {self.cash:.2f}"
        )

        self._update_equity()

    def _close_position(self, symbol: str) -> None:
        quantity = self.positions.get(symbol, 0.0)
        if quantity == 0:
            return

        direction = OrderDirection.SELL if quantity > 0 else OrderDirection.BUY
        order = OrderEvent(
            symbol     = symbol,
            order_type = OrderType.MARKET,
            direction  = direction,
            quantity   = abs(quantity),
        )
        self.events.put(order)

    def _update_equity(self) -> None:
        market_value = 0.0

        for symbol, qty in self.positions.items():
            if qty == 0:
                continue
            try:
                latest = self.data.get_latest(symbol, n=1)
                price  = float(latest["close"].iloc[-1])

                if qty > 0:
                    market_value += qty * price
                else:
                    entry = self.avg_price.get(symbol, price)
                    market_value += (entry - price) * abs(qty)

            except Exception:
                pass

        total_equity = self.cash + market_value

        if total_equity > self.initial_capital * 0.3:
            self.equity_curve.append({
                "timestamp"    : datetime.utcnow(),
                "cash"         : self.cash,
                "market_value" : market_value,
                "total_equity" : total_equity,
            })

    def compute_metrics(self) -> None:
        if not self.equity_curve:
            logger.warning("No equity curve data to compute metrics from.")
            return

        df      = pd.DataFrame(self.equity_curve)
        equity  = df["total_equity"]
        returns = equity.pct_change().dropna()

        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital
        ann_return   = (1 + total_return) ** (252 / max(len(returns), 1)) - 1
        ann_vol      = returns.std() * np.sqrt(252)
        sharpe       = ann_return / ann_vol if ann_vol != 0 else 0.0

        rolling_max  = equity.cummax()
        drawdown     = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        calmar       = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

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