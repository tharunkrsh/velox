"""
execution.py - Simulated execution handler for VELOX.

In backtesting, orders cannot be filled at exact prices —
markets have friction. This module models that friction:

    Slippage:    price moves against you when you trade
                 (you buy slightly higher, sell slightly lower)
    Commission:  brokerage fee per trade

In live trading, this gets replaced by a real broker adapter
(Alpaca, Interactive Brokers etc.) — zero other code changes.
"""

import logging

from core.events import (
    OrderEvent, FillEvent,
    OrderDirection
)

logger = logging.getLogger(__name__)


class SimulatedExecutionHandler:

    def __init__(
        self,
        data_handler,
        slippage_pct:   float = 0.001,   # 0.1% slippage per trade
        commission_pct: float = 0.001,   # 0.1% commission per trade
    ):
        self.data           = data_handler
        self.slippage_pct   = slippage_pct
        self.commission_pct = commission_pct
        self.events         = None  # injected by engine

    def on_order(self, event: OrderEvent) -> None:
        """
        Receive an OrderEvent, simulate execution,
        and fire a FillEvent back into the queue.
        """
        symbol   = event.symbol
        quantity = event.quantity
        direction = event.direction

        # Get the latest close price as our base fill price
        latest = self.data.get_latest(symbol, n=1)

        if latest.empty:
            logger.warning(f"No price data for {symbol}, order rejected.")
            return

        base_price = float(latest["close"].iloc[-1])

        # Apply slippage — buying costs more, selling earns less
        if direction == OrderDirection.BUY:
            fill_price = base_price * (1 + self.slippage_pct)
        else:
            fill_price = base_price * (1 - self.slippage_pct)

        # Calculate commission
        commission = quantity * fill_price * self.commission_pct

        fill = FillEvent(
            symbol     = symbol,
            direction  = direction,
            quantity   = quantity,
            fill_price = round(fill_price, 4),
            commission = round(commission, 4),
            exchange   = "SIM",
        )

        logger.debug(
            f"Fill: {direction.value} {quantity} {symbol} "
            f"@ {fill_price:.4f} (slippage: {self.slippage_pct*100:.1f}%, "
            f"commission: {commission:.2f})"
        )

        self.events.put(fill)