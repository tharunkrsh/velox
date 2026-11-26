"""Pure performance statistics over trade and day P&L (salvaged from the legacy
tearsheet, minus annualisation, which is meaningless for a challenge).

All functions take plain arrays and return floats; no I/O, no state.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel


def equity_curve(pnl: np.ndarray, initial: float) -> np.ndarray:
    """Balance after each P&L item, prefixed with the initial balance."""
    return initial + np.concatenate([[0.0], np.cumsum(np.asarray(pnl, dtype=float))])


def drawdowns(equity: np.ndarray) -> np.ndarray:
    """Absolute drawdown from running peak at each point (<= 0)."""
    equity = np.asarray(equity, dtype=float)
    return equity - np.maximum.accumulate(equity)


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough fall, as a positive amount."""
    return float(-drawdowns(equity).min()) if len(equity) else 0.0


def max_drawdown_pct(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=float)
    if not len(equity):
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(-((equity - peak) / peak).min())


def avg_drawdown(equity: np.ndarray) -> float:
    dd = drawdowns(equity)
    under = dd[dd < 0]
    return float(-under.mean()) if len(under) else 0.0


def _tail_q(level: float) -> float:
    return round(1.0 - level, 12)  # 1 - 0.8 == 0.19999999999999996 would shift the index


def var(pnl: np.ndarray, level: float = 0.95) -> float:
    """Historical value-at-risk as a positive loss amount (0 if the tail is a gain)."""
    pnl = np.asarray(pnl, dtype=float)
    if not len(pnl):
        return 0.0
    return float(max(0.0, -np.quantile(pnl, _tail_q(level), method="lower")))


def cvar(pnl: np.ndarray, level: float = 0.95) -> float:
    """Expected shortfall: mean loss in the worst (1 - level) tail, positive amount."""
    pnl = np.asarray(pnl, dtype=float)
    if not len(pnl):
        return 0.0
    cut = np.quantile(pnl, _tail_q(level), method="lower")
    return float(max(0.0, -pnl[pnl <= cut].mean()))


def win_rate(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    return float((pnl > 0).mean()) if len(pnl) else 0.0


def profit_factor(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def longest_losing_streak(pnl: np.ndarray) -> int:
    best = cur = 0
    for p in np.asarray(pnl, dtype=float):
        cur = cur + 1 if p < 0 else 0
        best = max(best, cur)
    return best


class PnlSummary(BaseModel):
    count: int
    total: float
    mean: float
    std: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    best: float
    worst: float
    var_95: float
    cvar_95: float
    longest_losing_streak: int


def summarize(pnl: np.ndarray) -> PnlSummary:
    pnl = np.asarray(pnl, dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return PnlSummary(
        count=len(pnl),
        total=float(pnl.sum()),
        mean=float(pnl.mean()) if len(pnl) else 0.0,
        std=float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        win_rate=win_rate(pnl),
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        profit_factor=profit_factor(pnl),
        best=float(pnl.max()) if len(pnl) else 0.0,
        worst=float(pnl.min()) if len(pnl) else 0.0,
        var_95=var(pnl),
        cvar_95=cvar(pnl),
        longest_losing_streak=longest_losing_streak(pnl),
    )
