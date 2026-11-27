from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hmmlearn")

from velox.sim.regime import InsufficientData, fit_regimes  # noqa: E402


def _two_state_series(seed: int = 0, n: int = 300):
    """Persistent Markov chain: cold days ~ N(-200, 300), hot days ~ N(+250, 120)."""
    rng = np.random.default_rng(seed)
    p_stay = 0.95
    states = np.zeros(n, dtype=int)
    for i in range(1, n):
        states[i] = states[i - 1] if rng.random() < p_stay else 1 - states[i - 1]
    pnl = np.where(states == 0, rng.normal(-200, 300, n), rng.normal(250, 120, n))
    return pnl, states


def test_recovers_hidden_states_with_worst_state_labelled_zero():
    pnl, truth = _two_state_series()
    m = fit_regimes(pnl, n_states=2, seed=7)
    assert m.means[0] < m.means[1]
    assert (m.states == truth).mean() > 0.85
    np.testing.assert_allclose(m.transmat.sum(axis=1), 1.0)
    assert np.all(np.diag(m.transmat) > 0.8)  # persistence recovered


def test_labels_stable_across_refits_and_seeds():
    pnl, _ = _two_state_series(seed=3)
    a = fit_regimes(pnl, seed=1)
    b = fit_regimes(pnl, seed=99)
    assert (a.states == b.states).mean() > 0.95


def test_refuses_small_samples():
    with pytest.raises(InsufficientData):
        fit_regimes(np.random.default_rng(0).normal(size=40))
