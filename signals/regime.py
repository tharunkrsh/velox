"""
regime.py - HMM-based market regime detector for VELOX.

Uses a Gaussian Hidden Markov Model to infer the current
market regime from price history.

Observable features (what we can measure):
    - Daily returns
    - Rolling volatility

Hidden states (what we're inferring):
    - Bull:     positive returns, low volatility
    - Sideways: flat returns, medium volatility
    - Bear:     negative returns, high volatility

Physics connection:
    Identical mathematics to quantum state inference —
    you never directly observe the state, only noisy
    measurements of it. Baum-Welch (the fitting algorithm)
    is essentially an EM algorithm over hidden state sequences,
    analogous to density matrix estimation in quantum mechanics.

Once fitted, the regime is inferred on every new bar and
broadcast to all strategies so they can adapt their behaviour.
"""

import logging
import numpy as np
import pandas as pd
from hmmlearn import hmm

from core.events import MarketEvent

logger = logging.getLogger(__name__)


# Regime labels — assigned after fitting based on state characteristics
BULL     = "bull"
SIDEWAYS = "sideways"
BEAR     = "bear"


class RegimeDetector:

    def __init__(
        self,
        data_handler,
        symbol:         str   = "AAPL",   # reference asset for regime
        n_states:       int   = 3,        # bull, sideways, bear
        lookback:       int   = 252,      # bars of history to fit on
        retrain_every:  int   = 63,       # refit every quarter
        min_train_bars: int   = 126,      # minimum bars before first fit
        vol_window:     int   = 20,       # rolling volatility window
    ):
        self.data           = data_handler
        self.symbol         = symbol
        self.n_states       = n_states
        self.lookback       = lookback
        self.retrain_every  = retrain_every
        self.min_train_bars = min_train_bars
        self.vol_window     = vol_window
        self.events         = None

        self.model          = None
        self.state_map      = {}    # HMM state index → regime label
        self.current_regime = None  # current inferred regime
        self.regime_history = []    # list of (timestamp, regime, probs)
        self.bar_count      = 0

    def on_market(self, event: MarketEvent) -> None:
        self.bar_count += 1

        # Refit model periodically
        if (self.bar_count >= self.min_train_bars and
                self.bar_count % self.retrain_every == 0):
            self._fit()

        # Infer current regime
        if self.model is not None:
            self._infer(event.timestamp)

    # ─── Model Fitting ────────────────────────────────────────────────────────

    def _fit(self) -> None:
        """
        Fit the Gaussian HMM on recent price history.
        Assigns regime labels based on learned state characteristics.
        """
        features = self._build_features(n=self.lookback)

        if features is None or len(features) < 50:
            return

        model = hmm.GaussianHMM(
            n_components    = self.n_states,
            covariance_type = "diag",
            n_iter          = 200,
            random_state    = 42,
            )


        try:
            model.fit(features)
            self.model = model
            self.state_map = self._assign_labels()
            logger.info(
                f"HMM fitted on {len(features)} bars. "
                f"Regimes: {self.state_map}"
            )
        except Exception as e:
            logger.warning(f"HMM fitting failed: {e}")

    def _assign_labels(self) -> dict:
        """
        After fitting, assign human-readable labels to states.
        We sort states by their mean return:
            highest mean return → bull
            lowest mean return  → bear
            middle              → sideways
        """
        # Mean return for each state (first feature = return)
        mean_returns = self.model.means_[:, 0]
        sorted_states = np.argsort(mean_returns)  # low to high

        return {
            int(sorted_states[0]): BEAR,
            int(sorted_states[1]): SIDEWAYS,
            int(sorted_states[2]): BULL,
        }

    # ─── Regime Inference ─────────────────────────────────────────────────────

    def _infer(self, timestamp) -> None:
        """
        Infer the current regime from recent data.
        Updates self.current_regime and appends to history.
        """
        features = self._build_features(n=self.vol_window + 5)

        if features is None or len(features) < 2:
            return

        try:
            # Get state probabilities for the most recent observation
            log_probs = self.model.predict_proba(features)
            probs     = log_probs[-1]   # probabilities for latest bar
            state     = np.argmax(probs)
            regime    = self.state_map.get(int(state), SIDEWAYS)

            self.current_regime = regime

            self.regime_history.append({
                "timestamp"   : timestamp,
                "regime"      : regime,
                "prob_bull"   : round(float(probs[self.state_map_inverse(BULL)]), 3),
                "prob_sideways": round(float(probs[self.state_map_inverse(SIDEWAYS)]), 3),
                "prob_bear"   : round(float(probs[self.state_map_inverse(BEAR)]), 3),
            })

            logger.debug(f"Regime: {regime} (probs: {np.round(probs, 2)})")

        except Exception as e:
            logger.warning(f"Regime inference failed: {e}")

    def state_map_inverse(self, label: str) -> int:
        """Get state index from label."""
        for k, v in self.state_map.items():
            if v == label:
                return k
        return 0

    # ─── Feature Engineering ──────────────────────────────────────────────────

    def _build_features(self, n: int) -> np.ndarray:
        """
        Build observation matrix for the HMM.
        Features: [daily_return, rolling_volatility]
        """
        bars = self.data.get_latest(self.symbol, n=n + self.vol_window)

        if bars.empty or len(bars) < self.vol_window + 2:
            return None

        close   = bars["close"].astype(float)
        returns = close.pct_change().dropna()
        vol     = returns.rolling(self.vol_window).std().dropna()

        # Align
        min_len  = min(len(returns), len(vol))
        returns  = returns.iloc[-min_len:]
        vol      = vol.iloc[-min_len:]

        features = np.column_stack([returns.values, vol.values])
        return features[-n:] if len(features) >= n else features

    # ─── Public Interface ─────────────────────────────────────────────────────

    def get_regime(self) -> str:
        """Returns current regime label or None if not yet fitted."""
        return self.current_regime

    def get_regime_df(self) -> pd.DataFrame:
        """Returns full regime history as a DataFrame."""
        if not self.regime_history:
            return pd.DataFrame()
        return pd.DataFrame(self.regime_history)