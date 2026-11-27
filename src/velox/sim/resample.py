"""Resampling historical trading days into simulated day sequences.

The unit is a whole trading day (keeping intraday trade order and clustering).

* :func:`stationary_bootstrap` (Politis & Romano 1994): blocks of consecutive
  historical days with geometric lengths (mean ``block_len``), wrapping around
  the end, which preserves short-run streaks and autocorrelation.
* :func:`regime_bootstrap`: a Markov chain over HMM regimes fitted to the
  trader's daily P&L. Each simulated day draws its state from the fitted
  transition matrix, then a historical day from that state.
"""

from __future__ import annotations

import numpy as np

from velox.sim.regime import RegimeModel


def stationary_bootstrap(
    n_days: int, n_paths: int, horizon: int, *, block_len: float = 4.0,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_days < 1:
        raise ValueError("need at least one historical trading day")
    if block_len < 1:
        raise ValueError("block_len must be >= 1")
    out = np.empty((n_paths, horizon), dtype=np.int64)
    out[:, 0] = rng.integers(0, n_days, n_paths)
    p_new = 1.0 / block_len
    for t in range(1, horizon):
        jump = rng.random(n_paths) < p_new
        nxt = (out[:, t - 1] + 1) % n_days
        out[:, t] = np.where(jump, rng.integers(0, n_days, n_paths), nxt)
    return out


def regime_bootstrap(
    model: RegimeModel, n_paths: int, horizon: int, *, rng: np.random.Generator,
) -> np.ndarray:
    pools = [np.flatnonzero(model.states == s) for s in range(model.n_states)]
    if any(len(p) == 0 for p in pools):
        raise ValueError("a regime has no historical days")
    cum_t = np.cumsum(model.transmat, axis=1)
    # start from the stationary distribution implied by the historical labels
    start = np.bincount(model.states, minlength=model.n_states) / len(model.states)
    state = np.searchsorted(np.cumsum(start), rng.random(n_paths), side="right")
    state = np.minimum(state, model.n_states - 1)
    out = np.empty((n_paths, horizon), dtype=np.int64)
    for t in range(horizon):
        if t:
            u = rng.random(n_paths)
            state = np.minimum((u[:, None] > cum_t[state]).sum(axis=1), model.n_states - 1)
        for s, pool in enumerate(pools):
            m = state == s
            out[m, t] = pool[rng.integers(0, len(pool), int(m.sum()))]
    return out
