"""
ml_signal.py - LightGBM-based alpha signal for VELOX.

Constructs features from price/volume history and trains
a gradient boosted classifier to predict next-bar direction.

Feature set:
    - Multi-timeframe returns (1, 5, 10, 20 bar)
    - Rolling volatility (10, 20 bar)
    - Volume ratio (current vs average)
    - RSI (14 bar)
    - Distance from moving averages (10, 20, 50 bar)

Target:
    Binary — will next bar's return be positive (1) or negative (0)?

Training regime:
    Walk-forward — model is retrained every `retrain_every` bars
    using only historical data up to that point.
    Never uses future data.

Why LightGBM over neural networks?
    Financial data is low signal-to-noise. Tree-based models
    handle tabular data better, are more interpretable,
    train faster, and are less prone to overfitting on small samples.
    LightGBM is the dominant model in quant ML competitions.
"""

import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import List

from core.events import MarketEvent, SignalEvent, SignalDirection

logger = logging.getLogger(__name__)


class MLSignalStrategy:

    def __init__(
        self,
        data_handler,
        symbols:        List[str],
        lookback:       int = 252,      # bars of history to train on
        retrain_every:  int = 21,       # retrain model every N bars (~ monthly)
        min_train_bars: int = 126,      # minimum bars before first training
        threshold:      float = 0.6,   # min predicted probability to trade
    ):
        self.data           = data_handler
        self.symbols        = symbols
        self.lookback       = lookback
        self.retrain_every  = retrain_every
        self.min_train_bars = min_train_bars
        self.threshold      = threshold
        self.events         = None

        self.models  = {}   # symbol → trained LightGBM model
        self.bar_count = 0

    def on_market(self, event: MarketEvent) -> None:
        self.bar_count += 1

        # Retrain models periodically
        if (self.bar_count >= self.min_train_bars and
                self.bar_count % self.retrain_every == 0):
            for symbol in self.symbols:
                self._train(symbol)

        # Generate signals if models exist
        if self.models:
            for symbol in self.symbols:
                self._predict(symbol)

    # ─── Feature Engineering ──────────────────────────────────────────────────

    def _build_features(self, symbol: str, n: int) -> pd.DataFrame:
        """
        Build feature matrix from price history.
        All features are constructed from data available at each bar —
        no lookahead bias possible.
        """
        bars = self.data.get_latest(symbol, n=n + 60)  # extra for rolling calcs

        if len(bars) < 60:
            return pd.DataFrame()

        close  = bars["close"].astype(float)
        volume = bars["volume"].astype(float) if "volume" in bars.columns else None

        features = pd.DataFrame(index=bars.index)

        # ── Returns ───────────────────────────────────────────────────────────
        for period in [1, 5, 10, 20]:
            features[f"ret_{period}"] = close.pct_change(period)

        # ── Volatility ────────────────────────────────────────────────────────
        daily_returns = close.pct_change()
        for period in [10, 20]:
            features[f"vol_{period}"] = daily_returns.rolling(period).std()

        # ── Distance from moving averages ─────────────────────────────────────
        for period in [10, 20, 50]:
            ma = close.rolling(period).mean()
            features[f"ma_dist_{period}"] = (close - ma) / ma

        # ── RSI (14 bar) ──────────────────────────────────────────────────────
        features["rsi_14"] = self._rsi(close, 14)

        # ── Volume ratio ──────────────────────────────────────────────────────
        if volume is not None:
            vol_ma = volume.rolling(20).mean()
            features["vol_ratio"] = volume / vol_ma.replace(0, np.nan)

        # Drop NaN rows from rolling calculations
        features = features.dropna()

        return features.tail(n)

    def _rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index — measures overbought/oversold conditions."""
        delta  = prices.diff()
        gain   = delta.clip(lower=0).rolling(period).mean()
        loss   = (-delta.clip(upper=0)).rolling(period).mean()
        rs     = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    # ─── Training ─────────────────────────────────────────────────────────────

    def _train(self, symbol: str) -> None:
        """
        Train a LightGBM classifier on historical features.

        Target: 1 if next bar return > 0, else 0
        Uses walk-forward — only trains on past data.
        """
        features = self._build_features(symbol, n=self.lookback)

        if len(features) < 50:
            return

        close = self.data.get_latest(symbol, n=self.lookback + 60)["close"]
        close = close.astype(float)

        # Forward return — what we're trying to predict
        # Shift by -1: at each bar, target is NEXT bar's return
        fwd_return = close.pct_change().shift(-1)

        # Align features and target
        aligned = features.join(fwd_return.rename("target")).dropna()

        if len(aligned) < 50:
            return

        X = aligned.drop(columns=["target"])
        y = (aligned["target"] > 0).astype(int)  # binary: up or down

        # Train LightGBM
        train_data = lgb.Dataset(X, label=y)

        params = {
            "objective":    "binary",
            "metric":       "binary_logloss",
            "num_leaves":   15,       # small tree = less overfitting
            "learning_rate": 0.05,
            "n_estimators": 100,
            "min_child_samples": 20,  # need at least 20 samples per leaf
            "verbose":      -1,       # suppress output
        }

        self.models[symbol] = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(period=-1)],
        )

        logger.info(f"ML model retrained for {symbol} on {len(X)} samples")

    # ─── Prediction ───────────────────────────────────────────────────────────

    def _predict(self, symbol: str) -> None:
        """
        Generate a signal based on model prediction.
        Only trades when model is confident (prob > threshold).
        """
        if symbol not in self.models:
            return

        features = self._build_features(symbol, n=1)

        if features.empty:
            return

        # Get probability of positive return
        prob = self.models[symbol].predict(features.values)[0]

        if prob > self.threshold:
            # Model confident price will go up
            signal = SignalEvent(
                symbol    = symbol,
                strategy  = "ml_signal",
                direction = SignalDirection.LONG,
                strength  = (prob - 0.5) * 2,  # scale 0.5-1.0 → 0.0-1.0
                metadata  = {"ml_prob": round(prob, 3)},
            )
            self.events.put(signal)

        elif prob < (1 - self.threshold):
            # Model confident price will go down
            signal = SignalEvent(
                symbol    = symbol,
                strategy  = "ml_signal",
                direction = SignalDirection.EXIT,
                strength  = 1.0,
                metadata  = {"ml_prob": round(prob, 3)},
            )
            self.events.put(signal)