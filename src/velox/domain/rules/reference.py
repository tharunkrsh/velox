"""Reference (single-path, plain Python) rule engine.

This is the readable source of truth for rule semantics. ``velox replay`` uses
it, and it is the oracle that the vectorised Monte Carlo engine must match
outcome-for-outcome (see ``tests/unit/test_engine_parity.py``).

Per trade, equity is modelled as falling from the pre-trade balance to a trough
(``balance - adverse``) and then settling at ``balance + pnl``. Floor rules that
include floating P&L are checked against the trough; the first floor hit as
equity falls is the one with the highest level. A position that hits a floor is
assumed liquidated exactly at that floor.

Per day: reset the daily-loss reference, run trades, then update the EOD
trailing peak. Evaluation pass conditions are checked after every trade (when
the phase has ``stop_on_target``) and at the end of each day. Passing a phase
starts the next one on a fresh account the following trading day. After the
last evaluation phase, the optional funded stage runs for ``funded_days``
trading days, taking payouts at day ends under its payout policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

import numpy as np

from velox.domain.outcomes import FAIL_OUTCOME, FailReason, FundedOutcome, Outcome
from velox.domain.rules.spec import Account, PayoutPolicy, PhaseSpec, RulesetSpec


@dataclass
class PathResult:
    outcome: Outcome
    phase_reached: int                 # index of the evaluation phase the path ended in
    days_used: int                     # evaluation trading days consumed (all phases)
    phase_pass_days: list[int]         # cumulative day count at which each phase passed
    consistency_delay_days: int        # days between first hitting the target and passing
    fail_rule: str | None = None
    halted_days: int = 0               # evaluation days cut short by a halt_day rule
    eod_balances: list[float] = field(default_factory=list)  # per eval day, current phase
    # funded stage (only when the evaluation passed and a funded spec + horizon are given)
    funded_outcome: FundedOutcome | None = None
    funded_days_used: int = 0
    funded_fail_rule: str | None = None
    n_payouts: int = 0
    gross_payouts: float = 0.0
    trader_payouts: float = 0.0        # split x gross (fee refunds are added by the caller)


DayTrades = tuple[np.ndarray, np.ndarray, np.ndarray]  # (pnl, adverse, units) for one day


@dataclass
class _DayEnd:
    fail_rule: str | None = None
    fail_reason: FailReason | None = None
    halted: bool = False
    passed_intraday: bool = False


def _run_day(floors, phase: PhaseSpec | None, acct: Account, day: DayTrades) -> _DayEnd:
    pnl, adverse, units = day
    acct.day_start_balance = acct.balance
    acct.day_pnl = 0.0
    acct.traded_today = False

    for p, adv, u in zip(pnl, adverse, units, strict=True):
        if u <= 0:  # sizing rounded this trade away
            continue
        acct.traded_today = True
        start = acct.balance
        trough = start - adv
        settle = start + p

        hit = None  # (rule, liquidation balance)
        floating = [r for r in floors if r.includes_floating and r.breached(trough, acct)]
        if floating:
            rule = max(floating, key=lambda r: r.floor(acct))
            hit = (rule, float(rule.floor(acct)))
        else:
            closing = [r for r in floors if not r.includes_floating and r.breached(settle, acct)]
            if closing:
                hit = (max(closing, key=lambda r: r.floor(acct)), settle)

        halted = False
        if hit is None:
            acct.balance = settle
        else:
            rule, liquidated_at = hit
            acct.balance = liquidated_at
            if rule.effect == "fail":
                acct.day_pnl = acct.balance - acct.day_start_balance
                return _DayEnd(rule.type, rule.reason)
            halted = True  # halt_day: liquidated at the floor, locked out for the day

        acct.day_pnl = acct.balance - acct.day_start_balance
        acct.equity_peak = max(acct.equity_peak, acct.balance)

        if halted:
            # liquidation can still breach a deeper floor (e.g. max loss below DLL)
            fails = [r for r in floors if r.effect == "fail" and r.breached(acct.balance, acct)]
            if fails:
                rule = max(fails, key=lambda r: r.floor(acct))
                return _DayEnd(rule.type, rule.reason)
            return _DayEnd(halted=True)

        if phase is not None and phase.stop_on_target and phase.pass_ok(acct):
            return _DayEnd(passed_intraday=True)
    return _DayEnd()


def _close_day(acct: Account) -> None:
    if acct.traded_today:
        acct.best_closed = max(acct.best_closed, acct.day_pnl)
        acct.pos_closed += max(acct.day_pnl, 0.0)
        acct.trading_days += 1
        acct.traded_today = False
    acct.eod_peak = max(acct.eod_peak, acct.balance)


def _take_payout(policy: PayoutPolicy, acct: Account, cap: float | None) -> float:
    gross = float(policy.gross_request(acct.profit, cap))
    if gross <= 0 or gross < policy.min_amount:
        return 0.0
    acct.balance -= gross
    acct.payouts += 1
    if policy.reset_peaks_on_payout:
        acct.eod_peak = acct.equity_peak = acct.balance
    return gross


def run_path(
    ruleset: RulesetSpec,
    days: Iterable[DayTrades],
    *,
    max_days: int | None = None,
    funded_days: int | None = None,
    payout_cap: float | None = None,
) -> PathResult:
    """Run one path. ``days`` is consumed lazily; a caller can keep iterating it
    afterwards to start a new attempt on the following day."""
    it: Iterator[DayTrades] = iter(days)
    basis = ruleset.initial_balance
    phase_idx = 0
    phase = ruleset.phases[0]
    acct = Account.fresh(basis, basis)
    r = PathResult(Outcome.INCOMPLETE, 0, 0, [], 0)
    first_target_day: int | None = None

    # ── evaluation ──
    for day in it:
        r.days_used += 1
        end = _run_day(phase.floors, phase, acct, day)
        if end.fail_rule is not None:
            r.eod_balances.append(acct.balance)
            r.outcome, r.fail_rule = FAIL_OUTCOME[end.fail_reason], end.fail_rule
            r.phase_reached = phase_idx
            return r
        r.halted_days += end.halted
        _close_day(acct)
        r.eod_balances.append(acct.balance)

        if first_target_day is None and phase.target_met(acct):
            first_target_day = r.days_used
        if end.passed_intraday or phase.pass_ok(acct):
            r.phase_pass_days.append(r.days_used)
            if first_target_day is not None:
                r.consistency_delay_days += r.days_used - first_target_day
            first_target_day = None
            if phase_idx == len(ruleset.phases) - 1:
                r.outcome = Outcome.PASSED
                break
            phase_idx += 1
            phase = ruleset.phases[phase_idx]
            acct = Account.fresh(basis, basis)

        if max_days is not None and r.days_used >= max_days:
            r.outcome = Outcome.TIMEOUT
            break
    r.phase_reached = phase_idx

    if r.outcome is not Outcome.PASSED or ruleset.funded is None or not funded_days:
        return r

    # ── funded stage ──
    f = ruleset.funded
    acct = Account.fresh(ruleset.funded_start, basis)
    r.funded_outcome = FundedOutcome.INCOMPLETE
    days_since = wins_since = 0
    for day in it:
        r.funded_days_used += 1
        end = _run_day(f.floors, None, acct, day)
        if end.fail_rule is not None:
            r.funded_outcome, r.funded_fail_rule = FundedOutcome.BREACHED, end.fail_rule
            break
        traded, day_pnl = acct.traded_today, acct.day_pnl
        _close_day(acct)
        if traded:
            days_since += 1
            wins_since += int(day_pnl >= f.payout.winning_day_threshold)
        if days_since >= f.payout.min_days_between and wins_since >= f.payout.min_winning_days:
            gross = _take_payout(f.payout, acct, payout_cap)
            if gross > 0:
                r.n_payouts += 1
                r.gross_payouts += gross
                days_since = wins_since = 0
        if r.funded_days_used >= funded_days:
            r.funded_outcome = FundedOutcome.SURVIVED
            break
    r.trader_payouts = r.gross_payouts * f.payout.split
    return r
