"""Regime detection on the trader's own daily P&L (salvaged from the legacy
market-regime HMM).

Used by the regime-aware bootstrap: each historical trading day is labelled
with a hidden state ("cold" to "hot" streak regimes), and resampled paths follow
the fitted state transition matrix so losing streaks cluster the way they did
historically.

Changes vs legacy ``signals/regime.py``:
* features come from the trader's daily P&L, not AAPL prices;
* ``n_states`` is configurable (default 2: small samples cannot support 3);
* several seeded restarts with the best log-likelihood kept, for stable fits;
* states are re-ordered by mean P&L (state 0 = worst) and the transition matrix
  is permuted to match, so labels are deterministic across refits;
* refuses to fit on too little data (caller falls back to the plain bootstrap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class InsufficientData(ValueError):
    pass


@dataclass(frozen=True)
class RegimeModel:
    n_states: int
    states: np.ndarray      # per historical day, 0 = worst-mean state
    means: np.ndarray       # mean daily P&L per state (account currency)
    transmat: np.ndarray    # [from, to], rows sum to 1, in re-ordered label space
    start_prob: np.ndarray
    log_likelihood: float


def _features(daily_pnl: np.ndarray, vol_window: int) -> np.ndarray:
    pnl = np.asarray(daily_pnl, dtype=float)
    std = pnl.std()
    z = (pnl - pnl.mean()) / (std if std > 0 else 1.0)
    # trailing dispersion of z over vol_window days (expanding at the start)
    disp = np.array([z[max(0, i - vol_window + 1): i + 1].std() for i in range(len(z))])
    return np.column_stack([z, disp])


def fit_regimes(
    daily_pnl: np.ndarray,
    n_states: int = 2,
    *,
    seed: int = 42,
    n_restarts: int = 5,
    min_days: int = 60,
    vol_window: int = 5,
) -> RegimeModel:
    try:
        from hmmlearn import hmm
    except ImportError as e:  # optional extra
        raise ImportError("regime-aware resampling needs: pip install 'velox[regime]'") from e

    daily_pnl = np.asarray(daily_pnl, dtype=float)
    if len(daily_pnl) < max(min_days, 10 * n_states):
        raise InsufficientData(
            f"{len(daily_pnl)} trading days is too few to fit {n_states} regimes "
            f"(need >= {max(min_days, 10 * n_states)})"
        )
    X = _features(daily_pnl, vol_window)

    best, best_ll = None, -np.inf
    for r in range(n_restarts):
        model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag",
                                n_iter=200, random_state=seed + r)
        try:
            model.fit(X)
            ll = model.score(X)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if ll > best_ll:
            best, best_ll = model, ll
    if best is None:
        raise InsufficientData("HMM failed to converge on this trade history")

    raw_states = best.predict(X)
    raw_means = np.array([daily_pnl[raw_states == s].mean() if np.any(raw_states == s)
                          else -np.inf for s in range(n_states)])
    order = np.argsort(raw_means, kind="stable")      # new label i <- raw state order[i]
    relabel = np.empty(n_states, dtype=int)
    relabel[order] = np.arange(n_states)

    return RegimeModel(
        n_states=n_states,
        states=relabel[raw_states],
        means=raw_means[order],
        transmat=best.transmat_[np.ix_(order, order)],
        start_prob=best.startprob_[order],
        log_likelihood=float(best_ll),
    )
