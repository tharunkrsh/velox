"""
momentum.py - Time-series momentum strategy for VELOX.

Logic:
    - Look back N bars
    - If cumulative return > threshold: fire LONG signal
    - If cumulative return < 0: fire EXIT signal

Based on Jegadeesh & Titman (1993) momentum factor.
One of the most robust and well-documented alpha signals
in quantitative finance.
"""

import logging

from core.events import MarketEvent, SignalEvent, SignalDirection

logger = logging.getLogger(__name__)


class MomentumStrategy:

    def __init__(
        self,
        data_handler,
        symbols:        list,
        lookback:       int   = 20,    # bars to look back
        threshold:      float = 0.02,  # min return to trigger long (2%)
    ):
        self.data      = data_handler
        self.symbols   = symbols
        self.lookback  = lookback
        self.threshold = threshold
        self.events    = None  # injected by engine

    def on_market(self, event: MarketEvent) -> None:
        """
        Called on every new bar.
        Calculates momentum for each symbol and fires signals.
        """
        for symbol in self.symbols:
            self._calculate(symbol)

    def _calculate(self, symbol: str) -> None:
        """
        Calculate momentum for a single symbol.
        Fires a SignalEvent if conditions are met.
        """
        bars = self.data.get_latest(symbol, n=self.lookback + 1)

        # Need enough bars to calculate momentum
        if len(bars) < self.lookback + 1:
            return

        # Cumulative return over the lookback window
        start_price = float(bars["close"].iloc[0])
        end_price   = float(bars["close"].iloc[-1])

        if start_price == 0:
            return

        momentum = (end_price - start_price) / start_price

        # Determine signal direction
        if momentum > self.threshold:
            direction = SignalDirection.LONG
            strength  = min(momentum / 0.10, 1.0)  # scale to 0-1, cap at 10% return
        elif momentum < 0:
            direction = SignalDirection.EXIT
            strength  = 1.0
        else:
            return  # no signal

        signal = SignalEvent(
            symbol    = symbol,
            strategy  = "momentum",
            direction = direction,
            strength  = strength,
        )

        logger.debug(
            f"Momentum signal: {symbol} {direction.value} "
            f"(momentum={momentum:.2%}, strength={strength:.2f})"
        )

        self.events.put(signal)