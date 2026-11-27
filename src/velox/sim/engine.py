"""Vectorised Monte Carlo rule engine.

Runs every (multiplier, path) pair as one *lane*. The loop is time-major (over
simulated days, then over the j-th trade of the day) and every operation is
vectorised across lanes. Rule maths is shared with the reference engine: the
rule methods in :mod:`velox.domain.rules.spec` accept an :class:`Account` whose
fields are NumPy arrays, so both engines evaluate identical formulas. The
reference engine (:mod:`velox.domain.rules.reference`) is the oracle; parity
is enforced by ``tests/unit/test_engine_parity.py``.

Common random numbers: every multiplier replays the same resampled day
sequences, so differences between multipliers come from sizing alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from velox.domain.daybook import DayBook
from velox.domain.outcomes import (
    FAIL_OUTCOME,
    FUNDED_CODE,
    FUNDED_OUTCOMES,
    NONE,
    OUTCOME_CODE,
    OUTCOMES,
    FundedOutcome,
    Outcome,
)
from velox.domain.rules.spec import (
    FLOOR_CODE,
    FLOOR_TYPES,
    Account,
    IntradayTrailingMaxLoss,
    RulesetSpec,
)
from velox.domain.sizing import SizingModel

_O, _FO, _R, _NONE = OUTCOME_CODE, FUNDED_CODE, FLOOR_CODE, NONE
_FAIL_CODE = {reason: _O[o] for reason, o in FAIL_OUTCOME.items()}


@dataclass(frozen=True)
class LaneResults:
    """Per-lane results, shape (n_multipliers, n_paths) unless noted."""

    multipliers: np.ndarray
    outcome: np.ndarray            # index into OUTCOMES
    fail_rule: np.ndarray          # index into RULE_TYPES, -1 = none
    phase_reached: np.ndarray
    days_used: np.ndarray
    consistency_delay_days: np.ndarray
    halted_days: np.ndarray
    funded_outcome: np.ndarray     # index into FUNDED_OUTCOMES, -1 = not funded
    funded_fail_rule: np.ndarray
    funded_days_used: np.ndarray
    n_payouts: np.ndarray
    gross_payouts: np.ndarray
    trader_payouts: np.ndarray
    eval_balances: np.ndarray | None  # (n_recorded, max_eval_days) eval EOD balance, NaN after end

    def lane(self, m: int, p: int) -> dict[str, object]:
        """Decoded result of one lane, keyed like the reference engine's ``PathResult``."""
        def dec(table, code):
            return None if code == _NONE else table[code]

        return {
            "outcome": OUTCOMES[self.outcome[m, p]],
            "fail_rule": dec(FLOOR_TYPES, self.fail_rule[m, p]),
            "phase_reached": int(self.phase_reached[m, p]),
            "days_used": int(self.days_used[m, p]),
            "consistency_delay_days": int(self.consistency_delay_days[m, p]),
            "halted_days": int(self.halted_days[m, p]),
            "funded_outcome": dec(FUNDED_OUTCOMES, self.funded_outcome[m, p]),
            "funded_fail_rule": dec(FLOOR_TYPES, self.funded_fail_rule[m, p]),
            "funded_days_used": int(self.funded_days_used[m, p]),
            "n_payouts": int(self.n_payouts[m, p]),
            "gross_payouts": float(self.gross_payouts[m, p]),
            "trader_payouts": float(self.trader_payouts[m, p]),
        }


class _State:
    """Full-width lane state; ``view(idx)`` exposes a subset as an array Account."""

    FIELDS = ("initial", "basis", "balance", "day_start_balance", "eod_peak", "equity_peak",
              "day_pnl", "best_closed", "pos_closed")
    ALL = (*FIELDS, "traded_today", "trading_days", "payouts")
    MUTABLE = tuple(f for f in ALL if f not in ("initial", "basis"))

    def __init__(self, n: int, start: float, basis: float):
        for f in self.FIELDS:
            setattr(self, f, np.full(n, start, dtype=float))
        self.basis[:] = basis
        self.basis_value = float(basis)
        self.day_pnl[:] = 0.0
        self.best_closed[:] = -np.inf
        self.pos_closed[:] = 0.0
        self.traded_today = np.zeros(n, dtype=bool)
        self.trading_days = np.zeros(n, dtype=np.int64)
        self.payouts = np.zeros(n, dtype=np.int64)

    def reset(self, idx: np.ndarray, start: float) -> None:
        for f in ("initial", "balance", "day_start_balance", "eod_peak", "equity_peak"):
            getattr(self, f)[idx] = start
        self.day_pnl[idx] = 0.0
        self.best_closed[idx] = -np.inf
        self.pos_closed[idx] = 0.0
        self.traded_today[idx] = False
        self.trading_days[idx] = 0
        self.payouts[idx] = 0

    def view(self, idx: np.ndarray) -> _LazyLanes:
        """Read-only view for rule evaluation; fields are sliced on first access."""
        return _LazyLanes(self, idx)

    def gather(self, idx: np.ndarray, initial: float) -> _Lanes:
        """Copy a same-stage group into local arrays (``initial``/``basis`` as scalars)."""
        loc = _Lanes({f: getattr(self, f)[idx] for f in self.MUTABLE})
        loc.initial, loc.basis = initial, self.basis_value
        return loc

    def scatter(self, idx: np.ndarray, loc: _Lanes) -> None:
        for f in self.MUTABLE:
            getattr(self, f)[idx] = getattr(loc, f)


class _Lanes:
    """Array-valued account for a group of lanes, with the same derived
    properties as :class:`Account` so rule methods apply unchanged."""

    profit = Account.profit
    best_day_incl_today = Account.best_day_incl_today
    positive_days_profit_incl_today = Account.positive_days_profit_incl_today
    trading_days_incl_today = Account.trading_days_incl_today

    def __init__(self, fields: dict[str, np.ndarray]):
        self.__dict__.update(fields)


class _LazyLanes(_Lanes):
    def __init__(self, state: _State, idx: np.ndarray):  # noqa: D107
        self.__dict__["_state"], self.__dict__["_idx"] = state, idx

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        v = getattr(self._state, name)[self._idx]
        self.__dict__[name] = v
        return v


def _subset(a: _Lanes, ix: np.ndarray, **override) -> _Lanes:
    """Slice a local group; scalars (initial, basis) pass through unchanged."""
    fields = {k: (v[ix] if isinstance(v, np.ndarray) else v) for k, v in a.__dict__.items()}
    fields.update(override)
    return _Lanes(fields)


def _first_max_hit(floors, values, a: Account, mask_rule):
    """For each lane: index of the breached floor with the highest level (first on ties),
    or -1. ``mask_rule(r)`` selects which rules are eligible."""
    n = len(values)
    best = np.full(n, -np.inf)
    choice = np.full(n, _NONE)
    for i, r in enumerate(floors):
        if not mask_rule(r):
            continue
        f = np.broadcast_to(np.asarray(r.floor(a), dtype=float), (n,))
        hit = ((values <= f) if r.breach_on_touch else (values < f)) & (f > best)
        best = np.where(hit, f, best)
        choice = np.where(hit, i, choice)
    return choice, best


def _run_day_group(floors, phase, a: _Lanes, base, ntr, pnl_f, adv_f, live_f):
    """One trading day for a group of lanes in the same stage, on local arrays.

    Mirrors ``reference._run_day``; lanes stop for the day on fail, halt or an
    intraday pass (``stopped``). Returns the updated group and per-lane flags.
    """
    n = len(base)
    a.day_start_balance = a.balance.copy()
    a.day_pnl = np.zeros(n)
    a.traded_today = np.zeros(n, dtype=bool)
    stopped = np.zeros(n, dtype=bool)
    failed = np.zeros(n, dtype=bool)
    halted = np.zeros(n, dtype=bool)
    passed = np.zeros(n, dtype=bool)
    code = np.full(n, _NONE)
    rule_ix = np.full(n, _NONE)
    fail_rules = [i for i, r in enumerate(floors) if r.effect == "fail"]
    # lookup by hit index + 1 (index 0 = no hit)
    is_fail_tab = np.array([False] + [r.effect == "fail" for r in floors])
    is_halt_tab = np.array([False] + [r.effect != "fail" for r in floors])
    check_pass = phase is not None and phase.stop_on_target

    # Floors that cannot move intraday (static, daily, EOD trailing) are computed once per
    # day; intraday trailing floors depend on equity_peak and are recomputed per trade.
    # A cheap "any lane near a floor?" test lets most trade steps skip exact hit selection.
    def _max_floor(sel):
        m = np.full(n, -np.inf)
        for r in floors:
            if sel(r):
                m = np.maximum(m, r.floor(a))
        return m

    is_moving = [isinstance(r, IntradayTrailingMaxLoss) for r in floors]
    has_close = any(not r.includes_floating for r in floors)
    const_float = _max_floor(lambda r: r.includes_floating
                             and not isinstance(r, IntradayTrailingMaxLoss))
    const_close = _max_floor(lambda r: not r.includes_floating
                             and not isinstance(r, IntradayTrailingMaxLoss)) if has_close else None
    moving_float = [r for r, mv in zip(floors, is_moving, strict=True)
                    if mv and r.includes_floating]
    moving_close = [r for r, mv in zip(floors, is_moving, strict=True)
                    if mv and not r.includes_floating]

    for j in range(int(ntr.max()) if n else 0):
        act = (j < ntr) & ~stopped
        if not act.any():
            break
        tr = np.where(act, base + j, 0)
        live = act & live_f[tr]
        if not live.any():
            continue
        a.traded_today |= live
        bal = a.balance
        trough = bal - np.where(live, adv_f[tr], 0.0)
        settle = bal + np.where(live, pnl_f[tr], 0.0)

        mf = const_float
        for r in moving_float:
            mf = np.maximum(mf, r.floor(a))
        near = trough <= mf
        if has_close:
            mc = const_close
            for r in moving_close:
                mc = np.maximum(mc, r.floor(a))
            near |= settle <= mc
        near &= live

        eq_pre = a.equity_peak
        # default: every live trade settles normally
        a.balance = settle
        a.day_pnl = settle - a.day_start_balance
        a.equity_peak = np.maximum(eq_pre, settle)
        ok = live

        if near.any():
            # exact hit selection only for the (few) lanes at or below some floor,
            # evaluated on the pre-trade state as in the reference engine
            ix = np.flatnonzero(near)
            sub = _subset(a, ix, balance=bal[ix], equity_peak=eq_pre[ix])
            tr_ix, st_ix = trough[ix], settle[ix]
            hit, level = _first_max_hit(floors, tr_ix, sub, lambda r: r.includes_floating)
            liq = level
            if has_close:
                hit2, _ = _first_max_hit(floors, st_ix, sub, lambda r: not r.includes_floating)
                use2 = (hit < 0) & (hit2 >= 0)
                hit = np.where(use2, hit2, hit)
                liq = np.where(use2, st_ix, liq)
            new_bal = np.where(hit >= 0, liq, st_ix)
            is_fail = is_fail_tab[hit + 1]
            is_halt = is_halt_tab[hit + 1]
            for i in fail_rules:
                m_i = hit == i
                code[ix[m_i]] = _FAIL_CODE[floors[i].reason]
                rule_ix[ix[m_i]] = _R[floors[i].type]
            a.balance[ix] = new_bal
            a.day_pnl[ix] = new_bal - a.day_start_balance[ix]
            a.equity_peak[ix] = np.where(is_fail, eq_pre[ix], np.maximum(eq_pre[ix], new_bal))
            failed[ix[is_fail]] = True
            stopped[ix[is_fail]] = True
            ok = ok.copy()
            ok[ix[is_fail]] = False

            if is_halt.any():
                hx = ix[is_halt]
                hs = _subset(a, hx)
                deep, _ = _first_max_hit(floors, a.balance[hx], hs, lambda r: r.effect == "fail")
                for i in fail_rules:
                    m_i = deep == i
                    code[hx[m_i]] = _FAIL_CODE[floors[i].reason]
                    rule_ix[hx[m_i]] = _R[floors[i].type]
                failed[hx[deep >= 0]] = True
                halted[hx[deep < 0]] = True
                stopped[hx] = True
                ok[hx] = False

        if check_pass:
            # cheap gate on all lanes, exact pass conditions only where the target is met
            cand = ok & np.asarray(phase.target_met(a), dtype=bool)
            if cand.any():
                cx = np.flatnonzero(cand)
                done = np.asarray(phase.pass_ok(_subset(a, cx)), dtype=bool)
                passed[cx[done]] = True
                stopped[cx[done]] = True
    # close the day (reference ``_close_day``) for lanes that did not fail
    traded = a.traded_today & ~failed
    a.best_closed = np.where(traded, np.maximum(a.best_closed, a.day_pnl), a.best_closed)
    a.pos_closed = np.where(traded, a.pos_closed + np.maximum(a.day_pnl, 0.0), a.pos_closed)
    a.trading_days = a.trading_days + traded
    a.traded_today = np.zeros(n, dtype=bool)
    a.eod_peak = np.where(failed, a.eod_peak, np.maximum(a.eod_peak, a.balance))
    return a, failed, halted, passed, code, rule_ix, traded


def simulate(
    ruleset: RulesetSpec,
    book: DayBook,
    day_idx: np.ndarray,
    multipliers: np.ndarray,
    *,
    sizing: SizingModel | None = None,
    max_eval_days: int,
    funded_days: int = 0,
    payout_cap: float | None = None,
    record_multipliers: list[int] | None = None,
    chunk_lanes: int = 24_000,
) -> LaneResults:
    """Simulate all lanes. ``day_idx[p, t]`` is the historical day replayed by path p
    on simulated day t; it needs at least ``max_eval_days + funded_days`` columns.

    Lanes are independent, so paths are processed in blocks of about ``chunk_lanes``
    lanes to keep the working set cache-sized; results are identical for any block size.
    """
    M = len(multipliers)
    per = max(1, chunk_lanes // max(M, 1))
    parts = [
        _simulate_block(ruleset, book, day_idx[i:i + per], multipliers, sizing=sizing,
                        max_eval_days=max_eval_days, funded_days=funded_days,
                        payout_cap=payout_cap, record_multipliers=record_multipliers)
        for i in range(0, len(day_idx), per)
    ]
    if len(parts) == 1:
        return parts[0]

    def cat(name):
        return np.concatenate([getattr(r, name) for r in parts], axis=1)

    rec = None
    if record_multipliers:
        k = len(record_multipliers)
        rec = np.concatenate(
            [r.eval_balances.reshape(k, -1, max_eval_days) for r in parts], axis=1
        ).reshape(-1, max_eval_days)
    fields = {f: cat(f) for f in LaneResults.__dataclass_fields__
              if f not in ("multipliers", "eval_balances")}
    return LaneResults(multipliers=parts[0].multipliers, eval_balances=rec, **fields)


def _simulate_block(
    ruleset: RulesetSpec,
    book: DayBook,
    day_idx: np.ndarray,
    multipliers: np.ndarray,
    *,
    sizing: SizingModel | None,
    max_eval_days: int,
    funded_days: int,
    payout_cap: float | None,
    record_multipliers: list[int] | None,
) -> LaneResults:
    sizing = sizing or SizingModel()
    multipliers = np.asarray(multipliers, dtype=float)
    M, (P, n_cols) = len(multipliers), day_idx.shape
    horizon = max_eval_days + (funded_days if ruleset.funded is not None else 0)
    if n_cols < horizon:
        raise ValueError(f"day_idx has {n_cols} columns, need {horizon}")
    L, T = M * P, book.n_trades
    n_phases = len(ruleset.phases)
    funded = ruleset.funded if funded_days else None
    FUNDED, DONE = n_phases, n_phases + 1

    # per-multiplier scaled trades, flattened [m * T + trade]
    scaled = [book.scaled(k, sizing) for k in multipliers]
    pnl_f = np.concatenate([s[0] for s in scaled])
    adv_f = np.concatenate([s[1] for s in scaled])
    live_f = np.concatenate([s[2] > 0 for s in scaled])
    counts = np.diff(book.offsets)
    lane_m = np.repeat(np.arange(M), P)
    lane_p = np.tile(np.arange(P), M)
    trade_base = lane_m * T

    basis = ruleset.initial_balance
    st = _State(L, basis, basis)
    stage = np.zeros(L, dtype=np.int64)

    out_outcome = np.full(L, _O[Outcome.INCOMPLETE])
    out_fail_rule = np.full(L, _NONE)
    out_phase = np.zeros(L, dtype=np.int64)
    days_used = np.zeros(L, dtype=np.int64)
    delay = np.zeros(L, dtype=np.int64)
    halted_days = np.zeros(L, dtype=np.int64)
    first_target = np.full(L, _NONE)
    f_outcome = np.full(L, _NONE)
    f_fail_rule = np.full(L, _NONE)
    f_days = np.zeros(L, dtype=np.int64)
    n_payouts = np.zeros(L, dtype=np.int64)
    gross = np.zeros(L)
    days_since = np.zeros(L, dtype=np.int64)
    wins_since = np.zeros(L, dtype=np.int64)

    rec_lanes = (np.concatenate([np.arange(m * P, (m + 1) * P) for m in record_multipliers])
                 if record_multipliers else np.zeros(0, dtype=np.int64))
    rec = np.full((len(rec_lanes), max_eval_days), np.nan) if record_multipliers else None

    def stage_floors(s: int):
        return funded.floors if s == FUNDED else ruleset.phases[s].floors

    for t in range(horizon):
        active = np.flatnonzero(stage != DONE)
        if not len(active):
            break
        day = day_idx[lane_p[active], t]
        first = book.offsets[day]
        ntr = counts[day]

        failed = np.zeros(L, dtype=bool)
        halted = np.zeros(L, dtype=bool)
        passed_intraday = np.zeros(L, dtype=bool)
        fail_code = np.full(L, _NONE)
        fail_rule_today = np.full(L, _NONE)
        traded_flag = np.zeros(L, dtype=bool)

        act_stage = stage[active]
        for s in np.unique(act_stage):
            gsel = act_stage == s
            g = active[gsel]
            start = ruleset.funded_start if s == FUNDED else basis
            res = _run_day_group(
                stage_floors(int(s)), ruleset.phases[s] if s < n_phases else None,
                st.gather(g, start), trade_base[g] + first[gsel], ntr[gsel],
                pnl_f, adv_f, live_f)
            loc, g_failed, g_halted, g_passed, g_code, g_rule, g_traded = res
            st.scatter(g, loc)
            failed[g], halted[g], passed_intraday[g] = g_failed, g_halted, g_passed
            fail_code[g], fail_rule_today[g], traded_flag[g] = g_code, g_rule, g_traded

        # ── end of day (stats already closed per group) ──
        # evaluation lanes
        ev = active[stage[active] < n_phases]
        days_used[ev] += 1
        if rec is not None:
            in_rec = np.isin(rec_lanes, ev)
            col = days_used[rec_lanes[in_rec]] - 1
            rec[np.flatnonzero(in_rec), col] = st.balance[rec_lanes[in_rec]]
        ev_fail = ev[failed[ev]]
        out_outcome[ev_fail] = fail_code[ev_fail]
        out_fail_rule[ev_fail] = fail_rule_today[ev_fail]
        out_phase[ev_fail] = stage[ev_fail]
        stage[ev_fail] = DONE
        ev = ev[~failed[ev]]
        halted_days[ev] += halted[ev]

        ev_stage = stage[ev].copy()  # snapshot: a lane passes at most one phase per day
        for s in range(n_phases):
            ls = ev[ev_stage == s]
            if not len(ls):
                continue
            phase = ruleset.phases[s]
            a = st.view(ls)
            tm = np.asarray(phase.target_met(a), dtype=bool) & (first_target[ls] == _NONE)
            first_target[ls[tm]] = days_used[ls[tm]]
            ok = passed_intraday[ls] | np.asarray(phase.pass_ok(st.view(ls)), dtype=bool)
            pl = ls[ok]
            if not len(pl):
                continue
            had = first_target[pl] != _NONE
            delay[pl[had]] += days_used[pl[had]] - first_target[pl[had]]
            first_target[pl] = _NONE
            if s == n_phases - 1:
                out_outcome[pl] = _O[Outcome.PASSED]
                out_phase[pl] = s
                if funded is not None:
                    stage[pl] = FUNDED
                    st.reset(pl, ruleset.funded_start)
                    f_outcome[pl] = _FO[FundedOutcome.INCOMPLETE]
                else:
                    stage[pl] = DONE
            else:
                stage[pl] = s + 1
                st.reset(pl, basis)

        still = ev[stage[ev] < n_phases]
        to = still[days_used[still] >= max_eval_days]
        out_outcome[to] = _O[Outcome.TIMEOUT]
        out_phase[to] = stage[to]
        stage[to] = DONE

        # funded lanes (those that were funded at the start of this day)
        fl = active[stage[active] == FUNDED]
        fl = fl[f_outcome[fl] != _NONE]
        fl = fl[np.isin(fl, ev, invert=True)]  # exclude lanes that just entered funded today
        if len(fl):
            f_days[fl] += 1
            br = fl[failed[fl]]
            f_outcome[br] = _FO[FundedOutcome.BREACHED]
            f_fail_rule[br] = fail_rule_today[br]
            stage[br] = DONE
            fl = fl[~failed[fl]]
            pol = funded.payout
            trd = fl[traded_flag[fl]]
            days_since[trd] += 1
            wins_since[trd] += st.day_pnl[trd] >= pol.winning_day_threshold
            elig = fl[(days_since[fl] >= pol.min_days_between)
                      & (wins_since[fl] >= pol.min_winning_days)]
            if len(elig):
                g = np.asarray(pol.gross_request(st.balance[elig] - st.initial[elig], payout_cap),
                               dtype=float)
                pay = (g > 0) & (g >= pol.min_amount)
                pe, g = elig[pay], g[pay]
                st.balance[pe] -= g
                st.payouts[pe] += 1
                if pol.reset_peaks_on_payout:
                    st.eod_peak[pe] = st.balance[pe]
                    st.equity_peak[pe] = st.balance[pe]
                n_payouts[pe] += 1
                gross[pe] += g
                days_since[pe] = 0
                wins_since[pe] = 0
            fin = fl[f_days[fl] >= funded_days]
            f_outcome[fin] = _FO[FundedOutcome.SURVIVED]
            stage[fin] = DONE

    split = funded.payout.split if funded is not None else 0.0

    def sh(x):
        return x.reshape(M, P)

    return LaneResults(
        multipliers=multipliers, outcome=sh(out_outcome), fail_rule=sh(out_fail_rule),
        phase_reached=sh(out_phase), days_used=sh(days_used),
        consistency_delay_days=sh(delay), halted_days=sh(halted_days),
        funded_outcome=sh(f_outcome), funded_fail_rule=sh(f_fail_rule),
        funded_days_used=sh(f_days), n_payouts=sh(n_payouts), gross_payouts=sh(gross),
        trader_payouts=sh(gross * split), eval_balances=rec,
    )
