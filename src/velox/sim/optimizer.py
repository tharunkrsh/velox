"""Turn raw lane outcomes into probabilities, expected value and a sizing recommendation.

Net EV at multiplier k, per challenge attempt::

    EV(k) = E[net funded payouts | k] - challenge_fee - P(pass | k) * activation_fee

where net funded payouts = profit split x gross payouts (+ the challenge fee if the
firm refunds it with the first payout), and paths that fail the evaluation or
never get paid contribute 0. Payouts already received before a funded breach
are kept; profit left in the account at the end of the funded horizon is not
counted (conservative).

``breakeven_funded_payout = fee / P(pass) + activation_fee`` is the model-free
amount a funded account would need to pay out on average for the attempt to
break even. It depends only on the pass probability, not on the payout model.
"""

from __future__ import annotations

import numpy as np

from velox.domain.outcomes import FUNDED_CODE, OUTCOME_CODE, FundedOutcome, Outcome
from velox.domain.rules.spec import FLOOR_TYPES
from velox.report.schema import MultiplierStats, Prob, StressPoint
from velox.sim.engine import LaneResults

_PASSED = OUTCOME_CODE[Outcome.PASSED]
_DAILY = OUTCOME_CODE[Outcome.FAIL_DAILY_LOSS]
_MAX = OUTCOME_CODE[Outcome.FAIL_MAX_LOSS]
_TIMEOUT = OUTCOME_CODE[Outcome.TIMEOUT]
_BREACH = FUNDED_CODE[FundedOutcome.BREACHED]


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> Prob:
    """95% Wilson score interval: well behaved near 0 and 1, unlike the normal one."""
    if n == 0:
        return Prob(p=0.0, lo=0.0, hi=1.0, n=0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Prob(p=round(p, 6), lo=round(max(0.0, centre - half), 6),
                hi=round(min(1.0, centre + half), 6), n=n)


def _net_payouts(res: LaneResults, m: int, fee: float, refund: bool) -> np.ndarray:
    net = res.trader_payouts[m].astype(float)
    if refund:
        net = net + fee * (res.n_payouts[m] > 0)
    return net


def lane_ev(res: LaneResults, m: int, *, fee: float, activation_fee: float,
            refund: bool) -> np.ndarray:
    passed = res.outcome[m] == _PASSED
    return _net_payouts(res, m, fee, refund) - fee - activation_fee * passed


def multiplier_stats(res: LaneResults, m: int, *, fee: float, activation_fee: float,
                     refund: bool) -> MultiplierStats:
    out = res.outcome[m]
    n = len(out)
    passed = out == _PASSED
    n_pass = int(passed.sum())
    delayed = passed & (res.consistency_delay_days[m] > 0)
    days = res.days_used[m][passed]
    ev_l = lane_ev(res, m, fee=fee, activation_fee=activation_fee, refund=refund)
    ev = float(ev_l.mean())
    se = float(ev_l.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    p_pass = n_pass / n if n else 0.0
    fail_rules = {}
    for i, name in enumerate(FLOOR_TYPES):
        share = float(np.mean(res.fail_rule[m] == i))
        if share > 0:
            fail_rules[name] = round(share, 6)

    return MultiplierStats(
        k=float(res.multipliers[m]),
        p_pass=wilson(n_pass, n),
        p_fail_daily_loss=wilson(int((out == _DAILY).sum()), n),
        p_fail_max_loss=wilson(int((out == _MAX).sum()), n),
        p_timeout=wilson(int((out == _TIMEOUT).sum()), n),
        p_consistency_delay=wilson(int(delayed.sum()), n),
        mean_delay_days=(round(float(res.consistency_delay_days[m][delayed].mean()), 3)
                         if delayed.any() else None),
        days_to_pass_p50=float(np.median(days)) if n_pass else None,
        days_to_pass_p90=float(np.quantile(days, 0.9)) if n_pass else None,
        fail_rules=fail_rules,
        p_payout=wilson(int((res.n_payouts[m] > 0).sum()), n),
        p_funded_breach=wilson(int((res.funded_outcome[m] == _BREACH).sum()), n),
        mean_payouts=round(float(res.n_payouts[m].mean()), 4),
        mean_net_payout=round(float(_net_payouts(res, m, fee, refund).mean()), 2),
        ev=round(ev, 2),
        ev_lo=round(ev - 1.96 * se, 2),
        ev_hi=round(ev + 1.96 * se, 2),
        breakeven_funded_payout=(round(fee / p_pass + activation_fee, 2) if p_pass > 0
                                 else None),
    )


def grid_stats(res: LaneResults, **kw) -> list[MultiplierStats]:
    return [multiplier_stats(res, m, **kw) for m in range(len(res.multipliers))]


def best_index(grid: list[MultiplierStats]) -> int:
    """Highest EV; ties go to the smaller multiplier (less risk for the same value)."""
    evs = np.array([g.ev for g in grid])
    return int(np.flatnonzero(evs == evs.max())[0])


def stress_point(g: MultiplierStats) -> StressPoint:
    return StressPoint(k=g.k, p_pass=g.p_pass, ev=g.ev)
