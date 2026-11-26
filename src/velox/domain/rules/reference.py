"""Reference (single-path, plain Python) rule engine.

This is the readable source of truth for rule semantics. It is used by
``velox replay`` and as the test oracle that the vectorised Monte Carlo engine
must match outcome-for-outcome.

Per trade, equity is modelled as falling from the pre-trade balance to a trough
(``balance - adverse``) and then settling at ``balance + pnl``. Floor rules that
include floating P&L are checked against the trough; the first floor hit as
equity falls is the one with the highest level. A position that hits a floor is
assumed liquidated exactly at that floor.

Per day: reset the daily-loss reference, run trades, then update the EOD
trailing peak. Pass conditions are checked after every trade (when the phase
has ``stop_on_target``) and at the end of each day. On passing a phase the next
phase starts on a fresh account the following trading day.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from velox.domain.rules.spec import (
    Account,
    FailReason,
    PhaseSpec,
    ProfitTarget,
    RulesetSpec,
)


class Outcome(StrEnum):
    PASSED = "passed"
    FAIL_DAILY_LOSS = "fail_daily_loss"
    FAIL_MAX_LOSS = "fail_max_loss"
    TIMEOUT = "timeout"        # hit the simulation's max-days cap
    INCOMPLETE = "incomplete"  # ran out of trade history (replay)


_FAIL = {FailReason.DAILY_LOSS: Outcome.FAIL_DAILY_LOSS, FailReason.MAX_LOSS: Outcome.FAIL_MAX_LOSS}


@dataclass
class PathResult:
    outcome: Outcome
    phase_reached: int                 # index of the phase the path ended in
    days_used: int                     # trading days consumed across all phases
    phase_pass_days: list[int]         # cumulative day count at which each phase passed
    consistency_delay_days: int        # days between first hitting the target and passing
    fail_rule: str | None = None
    halted_days: int = 0               # days cut short by a halt_day rule
    eod_balances: list[float] = field(default_factory=list)  # per day, current phase account


def _pass_ok(phase: PhaseSpec, a: Account) -> bool:
    return all(r.met(a) for r in phase.pass_conditions)


DayTrades = tuple[np.ndarray, np.ndarray, np.ndarray]  # (pnl, adverse, units) for one day


def run_path(
    ruleset: RulesetSpec,
    days: Iterable[DayTrades],
    *,
    max_days: int | None = None,
) -> PathResult:
    initial = ruleset.initial_balance
    phase_idx = 0
    phase = ruleset.phases[0]
    acct = Account.fresh(initial)
    pass_days: list[int] = []
    eod: list[float] = []
    first_target_day: int | None = None
    delay = 0
    halted_days = 0
    day_no = 0

    def result(outcome: Outcome, fail_rule: str | None = None) -> PathResult:
        return PathResult(outcome, phase_idx, day_no, pass_days, delay, fail_rule, halted_days, eod)

    for pnl, adverse, units in days:
        day_no += 1
        acct.day_start_balance = acct.balance
        acct.day_pnl = 0.0
        acct.traded_today = False
        passed_today = False
        halted = False

        for p, adv, u in zip(pnl, adverse, units, strict=True):
            if u <= 0:  # sizing rounded this trade away
                continue
            acct.traded_today = True
            start = acct.balance
            trough = start - adv
            settle = start + p

            hit = None  # (rule, liquidation balance)
            floating_hits = [r for r in phase.floors
                             if r.includes_floating and r.breached(trough, acct)]
            if floating_hits:
                rule = max(floating_hits, key=lambda r: r.floor(acct))
                hit = (rule, rule.floor(acct))
            else:
                closing_hits = [r for r in phase.floors if not r.includes_floating
                                and r.breached(settle, acct)]
                if closing_hits:
                    hit = (max(closing_hits, key=lambda r: r.floor(acct)), settle)

            if hit is None:
                acct.balance = settle
            else:
                rule, liquidated_at = hit
                if rule.effect == "fail":
                    acct.balance = liquidated_at
                    acct.day_pnl = acct.balance - acct.day_start_balance
                    eod.append(acct.balance)
                    return result(_FAIL[rule.reason], rule.type)
                # halt_day: liquidated at the floor, locked out for the rest of the day
                acct.balance = liquidated_at
                halted = True

            acct.day_pnl = acct.balance - acct.day_start_balance
            acct.equity_peak = max(acct.equity_peak, acct.balance)

            if halted:
                # liquidation can still breach a deeper floor (e.g. max loss below DLL)
                fails = [r for r in phase.floors
                         if r.effect == "fail" and r.breached(acct.balance, acct)]
                if fails:
                    rule = max(fails, key=lambda r: r.floor(acct))
                    eod.append(acct.balance)
                    return result(_FAIL[rule.reason], rule.type)
                halted_days += 1
                break

            if phase.stop_on_target and _pass_ok(phase, acct):
                passed_today = True
                break

        # ── end of day ──
        if acct.traded_today:
            acct.closed_day_pnls.append(acct.day_pnl)
            acct.trading_days += 1
            acct.traded_today = False
        acct.eod_peak = max(acct.eod_peak, acct.balance)
        eod.append(acct.balance)

        if first_target_day is None and _target_met(phase, acct):
            first_target_day = day_no
        if passed_today or _pass_ok(phase, acct):
            pass_days.append(day_no)
            if first_target_day is not None:
                delay += day_no - first_target_day
            first_target_day = None
            if phase_idx == len(ruleset.phases) - 1:
                return result(Outcome.PASSED)
            phase_idx += 1
            phase = ruleset.phases[phase_idx]
            acct = Account.fresh(initial)

        if max_days is not None and day_no >= max_days:
            return result(Outcome.TIMEOUT)

    return result(Outcome.INCOMPLETE)


def _target_met(phase: PhaseSpec, a: Account) -> bool:
    return all(r.met(a) for r in phase.pass_conditions if isinstance(r, ProfitTarget))
