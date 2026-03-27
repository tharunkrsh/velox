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
import numpy as np

from core.events import MarketEvent, SignalEvent, SignalDirection

logger = logging.getLogger(__name__)


class MomentumStrategy:

    def __init__(
        self,
        data_handler,
        symbols:        list,
        lookback:       int   = 20,    # bars to look back
        enter_threshold: float = 0.03,  # enter long only above +3%
        exit_threshold:  float = -0.01, # exit only below -1%
        min_hold_bars:   int   = 5,     # avoid immediate churn
        rebalance_every: int   = 5,     # evaluate every N bars
        trend_filter_ma: int   = 0,     # require price > MA for longs if > 0
        vol_target:      float = 0.0,   # annualized vol target (0 disables)
        vol_lookback:    int   = 20,    # bars for realized vol estimate
        max_vol_scale:   float = 1.5,   # cap leverage from vol scaling
        regime_detector       = None,
    ):
        self.data      = data_handler
        self.symbols   = symbols
        self.lookback  = lookback
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.min_hold_bars = max(0, int(min_hold_bars))
        self.rebalance_every = max(1, int(rebalance_every))
        self.trend_filter_ma = max(0, int(trend_filter_ma))
        self.vol_target = max(0.0, float(vol_target))
        self.vol_lookback = max(5, int(vol_lookback))
        self.max_vol_scale = max(0.1, float(max_vol_scale))
        self.events    = None  # injected by engine
        self.regime_detector  = regime_detector
        self.bar_count = 0
        self.entry_bar = {}

    def on_market(self, event: MarketEvent) -> None:
        """
        Called on every new bar.
        Calculates momentum for each symbol and fires signals.
        """
        self.bar_count += 1
        if self.bar_count % self.rebalance_every != 0:
            return

        for symbol in self.symbols:
            self._calculate(symbol)

    def _calculate(self, symbol: str) -> None:
        if self.regime_detector is not None:
            regime = self.regime_detector.get_regime()
            if regime == "bear":
                return

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
        if momentum > self.enter_threshold:
            if self.trend_filter_ma > 0:
                trend_bars = self.data.get_latest(symbol, n=self.trend_filter_ma)
                if len(trend_bars) < self.trend_filter_ma:
                    return
                ma = float(trend_bars["close"].astype(float).mean())
                if end_price <= ma:
                    return

            direction = SignalDirection.LONG
            strength  = min(momentum / 0.10, 1.0)  # base 0-1 scaling
            strength *= self._vol_scale(symbol)
            strength = min(max(strength, 0.0), 1.0)
        elif momentum < self.exit_threshold:
            entered_at = self.entry_bar.get(symbol)
            if entered_at is not None and (self.bar_count - entered_at) < self.min_hold_bars:
                return
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

        if direction == SignalDirection.LONG:
            self.entry_bar[symbol] = self.bar_count
        elif direction == SignalDirection.EXIT:
            self.entry_bar.pop(symbol, None)

        self.events.put(signal)

    def _vol_scale(self, symbol: str) -> float:
        if self.vol_target <= 0:
            return 1.0
        bars = self.data.get_latest(symbol, n=self.vol_lookback + 1)
        if len(bars) < self.vol_lookback + 1:
            return 1.0
        close = bars["close"].astype(float)
        returns = close.pct_change().dropna()
        if returns.empty:
            return 1.0
        realized_vol = float(returns.std() * np.sqrt(252))
        if realized_vol <= 1e-8:
            return 1.0
        scale = self.vol_target / realized_vol
        return min(max(scale, 0.25), self.max_vol_scale)