"""
pairs.py - Kalman filter pairs trading strategy for VELOX.

Uses a Kalman filter to dynamically estimate the hedge ratio
between two cointegrated assets. This is the industry standard
approach — far superior to static OLS which uses a fixed
lookback window and cannot adapt to regime changes.

The Kalman filter treats the hedge ratio as a latent variable
that evolves as a random walk, and updates our estimate
optimally (minimum variance) on each new observation.

Mathematical connection to physics:
    This is identical to tracking a particle's position
    given noisy measurements — the same math used in
    GPS, rocket guidance, and quantum state estimation.

State equation:    beta_t = beta_{t-1} + w_t   (hedge ratio drifts)
Observation:       y_t = beta_t * x_t + v_t    (prices are noisy)

Where w_t ~ N(0, Q) and v_t ~ N(0, R) are process/observation noise.
"""

import logging
import numpy as np
from typing import Tuple

from core.events import MarketEvent, SignalEvent, SignalDirection

logger = logging.getLogger(__name__)


class KalmanPairsStrategy:

    def __init__(
        self,
        data_handler,
        pair:          Tuple[str, str],
        z_score_entry: float = 2.0,
        z_score_exit:  float = 0.5,
        # Kalman filter parameters
        delta:         float = 1e-4,   # how fast hedge ratio can change
        vt:            float = 1e-3,   # observation noise variance
    ):
        self.data          = data_handler
        self.symbol_a      = pair[0]
        self.symbol_b      = pair[1]
        self.z_score_entry = z_score_entry
        self.z_score_exit  = z_score_exit
        self.events        = None

        # ── Kalman filter state ───────────────────────────────────────────────
        # We're estimating a 2D state: [hedge_ratio, intercept]
        # This lets the spread level (intercept) drift too, not just the slope

        self.delta = delta
        self.vt    = vt

        # Process noise covariance — how much can state change per bar
        self.Wt = delta / (1 - delta) * np.eye(2)

        # State estimate: [beta, alpha] = [hedge ratio, intercept]
        self.theta = np.zeros(2)

        # State covariance — our uncertainty about the state
        self.P = np.zeros((2, 2))

        # Running spread statistics for z-score
        self.spread_mean = 0.0
        self.spread_var  = 1.0
        self.n_obs       = 0

        # Position state
        self.position: str = None

        # Minimum bars before trading
        self.warmup = 30
        self.bar_count = 0

    def on_market(self, event: MarketEvent) -> None:
        self._calculate()

    def _calculate(self) -> None:
        bars_a = self.data.get_latest(self.symbol_a, n=1)
        bars_b = self.data.get_latest(self.symbol_b, n=1)

        if bars_a.empty or bars_b.empty:
            return

        price_a = float(bars_a["close"].iloc[-1])
        price_b = float(bars_b["close"].iloc[-1])

        # Update Kalman filter with new observation
        spread = self._kalman_update(price_a, price_b)

        self.bar_count += 1

        # Need warmup period before trading
        if self.bar_count < self.warmup:
            return

        # Update running spread statistics (online mean/variance)
        self.n_obs       += 1
        old_mean          = self.spread_mean
        self.spread_mean += (spread - self.spread_mean) / self.n_obs
        self.spread_var  += (spread - old_mean) * (spread - self.spread_mean)

        if self.n_obs < 2:
            return

        spread_std = np.sqrt(self.spread_var / (self.n_obs - 1))

        if spread_std == 0:
            return

        z_score = (spread - self.spread_mean) / spread_std

        logger.debug(
            f"Kalman pairs {self.symbol_a}/{self.symbol_b} "
            f"z={z_score:.2f} beta={self.theta[0]:.3f} "
            f"position={self.position}"
        )

        # ── Entry logic ───────────────────────────────────────────────────────

        if self.position is None:
            if z_score > self.z_score_entry:
                self._fire(self.symbol_a, SignalDirection.SHORT, abs(z_score) / 4)
                self._fire(self.symbol_b, SignalDirection.LONG,  abs(z_score) / 4)
                self.position = "short_spread"

            elif z_score < -self.z_score_entry:
                self._fire(self.symbol_a, SignalDirection.LONG,  abs(z_score) / 4)
                self._fire(self.symbol_b, SignalDirection.SHORT, abs(z_score) / 4)
                self.position = "long_spread"

        # ── Exit logic ────────────────────────────────────────────────────────

        elif self.position == "short_spread" and z_score < self.z_score_exit:
            self._fire(self.symbol_a, SignalDirection.EXIT, 1.0)
            self._fire(self.symbol_b, SignalDirection.EXIT, 1.0)
            self.position = None

        elif self.position == "long_spread" and z_score > -self.z_score_exit:
            self._fire(self.symbol_a, SignalDirection.EXIT, 1.0)
            self._fire(self.symbol_b, SignalDirection.EXIT, 1.0)
            self.position = None

    def _kalman_update(self, price_a: float, price_b: float) -> float:
        """
        Run one step of the Kalman filter.

        Observation model: price_a = theta[0] * price_b + theta[1] + noise
        We update our estimate of theta = [hedge_ratio, intercept]

        Returns the current spread (residual after hedge).
        """
        # Observation vector: [price_b, 1]
        # (the "features" that predict price_a)
        F = np.array([price_b, 1.0])

        # ── Predict step ──────────────────────────────────────────────────────
        # State drifts by process noise Wt
        P_pred = self.P + self.Wt

        # ── Update step ───────────────────────────────────────────────────────
        # Innovation (how wrong was our prediction?)
        y_pred      = F @ self.theta                 # predicted price_a
        innovation  = price_a - y_pred               # actual vs predicted

        # Innovation covariance
        S = F @ P_pred @ F.T + self.vt

        # Kalman gain — how much to trust the new observation
        K = P_pred @ F.T / S

        # Update state estimate
        self.theta = self.theta + K * innovation

        # Update covariance
        self.P = (np.eye(2) - np.outer(K, F)) @ P_pred

        # Spread = innovation = actual price_a minus Kalman-predicted price_a
        return innovation

    def _fire(self, symbol: str, direction: SignalDirection, strength: float) -> None:
        signal = SignalEvent(
            symbol    = symbol,
            strategy  = "kalman_pairs",
            direction = direction,
            strength  = min(strength, 1.0),
        )
        self.events.put(signal)