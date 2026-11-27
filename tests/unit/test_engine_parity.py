"""Vectorised engine == reference engine, outcome for outcome, on randomised
multi-stage paths (evaluation phases + funded stage), presets and random rulesets."""

from __future__ import annotations

import numpy as np
import pytest

from velox.domain.daybook import DayBook
from velox.domain.outcomes import FUNDED_OUTCOMES, OUTCOMES
from velox.domain.rules.reference import run_path
from velox.domain.rules.spec import RulesetSpec
from velox.domain.sizing import IntratradeModel, SizingModel
from velox.presets import list_presets
from velox.sim.engine import simulate
from velox.sim.resample import stationary_bootstrap


def _book(rng: np.random.Generator, scale: float, n_days: int = 40) -> DayBook:
    counts = rng.integers(1, 6, n_days)
    n = int(counts.sum())
    units = rng.integers(1, 4, n).astype(float)
    # coarse grid of values so trades land exactly on limits now and then
    pnl_pu = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5], n) * scale / 4
    adverse_pu = np.maximum(-pnl_pu, 0) + rng.choice([0, 0, 1, 2], n) * scale / 4
    days = np.arange(n_days).astype("datetime64[D]")
    return DayBook(days=days, offsets=np.concatenate([[0], np.cumsum(counts)]), units=units,
                   pnl_per_unit=pnl_pu, adverse_per_unit=adverse_pu,
                   intratrade=IntratradeModel.MAE)


def _random_ruleset(rng: np.random.Generator) -> RulesetSpec:
    def floor():
        kind = rng.choice(["static", "eod", "intraday", "daily"])
        amt = float(rng.choice([500, 800, 1000, 1500]))
        common = {"amount": amt, "breach_on_touch": bool(rng.random() < 0.7),
                  "includes_floating": bool(rng.random() < 0.8)}
        if kind == "static":
            return {"type": "static_max_loss", **common}
        if kind == "daily":
            return {"type": "daily_loss_limit", **common,
                    "effect": str(rng.choice(["fail", "halt_day"]))}
        lock = rng.choice([None, 0.0, 100.0])
        return {"type": "eod_trailing_max_loss" if kind == "eod" else "intraday_trailing_max_loss",
                **common, "lock_offset": None if lock is None else float(lock),
                "post_payout_lock_offset": float(rng.choice([0.0, 50.0])) if rng.random() < 0.4
                else None}

    def phase(i):
        conds = [{"type": "profit_target", "amount": float(rng.choice([600, 1000, 1500]))}]
        if rng.random() < 0.5:
            conds.append({"type": "min_trading_days", "days": int(rng.integers(1, 5))})
        if rng.random() < 0.6:
            conds.append({"type": "best_day_consistency", "max_share": float(rng.choice([.4, .5])),
                          "denominator": str(rng.choice(["total_profit", "positive_days_profit"])),
                          "strict": bool(rng.random() < 0.3)})
        return {"name": f"P{i}", "floors": [floor() for _ in range(int(rng.integers(1, 4)))],
                "pass_conditions": conds, "stop_on_target": bool(rng.random() < 0.7)}

    spec = {
        "id": "random", "name": "random", "version": "t", "source_urls": [],
        "verified_at": "2026-01-01", "initial_balance": 10000.0,
        "calendar": {"timezone": "UTC"},
        "phases": [phase(i) for i in range(int(rng.integers(1, 4)))],
        "funded": {
            "name": "F", "initial_balance": float(rng.choice([0.0, 10000.0])),
            "floors": [floor() for _ in range(int(rng.integers(1, 3)))],
            "payout": {"split": 0.8, "min_days_between": int(rng.integers(0, 4)),
                       "min_winning_days": int(rng.integers(0, 3)),
                       "winning_day_threshold": float(rng.choice([0, 100])),
                       "max_share": float(rng.choice([0.5, 1.0])),
                       "max_amount": float(rng.choice([400, 2000])) if rng.random() < .5 else None,
                       "reset_peaks_on_payout": bool(rng.random() < 0.7)},
        },
    }
    return RulesetSpec.model_validate(spec)


def _check(spec, book, multipliers, sizing, n_paths, max_eval_days, funded_days, payout_cap,
           seed):
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap(book.n_days, n_paths, max_eval_days + funded_days,
                               block_len=3, rng=rng)
    res = simulate(spec, book, idx, multipliers, sizing=sizing, max_eval_days=max_eval_days,
                   funded_days=funded_days, payout_cap=payout_cap,
                   record_multipliers=list(range(len(multipliers))))
    lanes = 0
    for m, k in enumerate(multipliers):
        pnl, adv, units = book.scaled(k, sizing)
        for p in range(n_paths):
            seq = [(pnl[book.day_slice(d)], adv[book.day_slice(d)], units[book.day_slice(d)])
                   for d in idx[p]]
            ref = run_path(spec, seq, max_days=max_eval_days, funded_days=funded_days,
                           payout_cap=payout_cap)
            got = res.lane(m, p)
            want = {k2: getattr(ref, k2) for k2 in got}
            assert got == pytest.approx(want), f"k={k} path={p}"
            row = res.eval_balances[m * n_paths + p]
            n_rec = len(ref.eod_balances)
            np.testing.assert_allclose(row[:n_rec], ref.eod_balances, err_msg=f"k={k} p={p}")
            assert np.isnan(row[n_rec:]).all()
            lanes += 1
    return lanes, res


@pytest.mark.parametrize("seed", range(8))
def test_parity_random_rulesets(seed):
    rng = np.random.default_rng(1000 + seed)
    spec = _random_ruleset(rng)
    book = _book(rng, scale=float(rng.choice([100, 200])))
    sizing = SizingModel(unit_step=1.0, max_units=4.0) if seed % 2 else SizingModel()
    lanes, res = _check(spec, book, np.array([0.5, 1.0, 1.75]), sizing, n_paths=60,
                        max_eval_days=25, funded_days=15, payout_cap=None if seed % 3 else 700.0,
                        seed=seed)
    assert lanes == 180


@pytest.mark.parametrize("preset", [p.id for p in list_presets()])
def test_parity_presets(preset):
    spec = next(p for p in list_presets() if p.id == preset)
    rng = np.random.default_rng(7)
    book = _book(rng, scale=spec.initial_balance / 40)
    lanes, res = _check(spec, book, np.array([0.5, 1.0, 2.0]), SizingModel(), n_paths=60,
                        max_eval_days=30, funded_days=20, payout_cap=None, seed=11)
    assert lanes == 180


def test_parity_covers_every_outcome_class():
    """Guard against a vacuous parity test: the randomized suite must exercise
    passes, both fail reasons, timeouts, funded breaches/survivals and payouts."""
    seen_o, seen_f, payouts, halts = set(), set(), 0, 0
    for seed in range(8):
        rng = np.random.default_rng(1000 + seed)
        spec = _random_ruleset(rng)
        book = _book(rng, scale=float(rng.choice([100, 200])))
        idx = stationary_bootstrap(book.n_days, 60, 40, block_len=3, rng=np.random.default_rng(0))
        r = simulate(spec, book, idx, np.array([0.5, 1.0, 1.75]), max_eval_days=25,
                     funded_days=15)
        seen_o |= {OUTCOMES[i] for i in np.unique(r.outcome)}
        seen_f |= {FUNDED_OUTCOMES[i] for i in np.unique(r.funded_outcome) if i >= 0}
        payouts += int(r.n_payouts.sum())
        halts += int(r.halted_days.sum())
    assert {"passed", "fail_daily_loss", "fail_max_loss", "timeout"} <= seen_o
    assert {"breached", "survived"} <= seen_f
    assert payouts > 0
    assert halts > 0


@pytest.mark.parametrize("seed", range(3))
def test_chunked_simulation_matches_single_block(seed):
    """Lanes are processed in blocks; results must not depend on the block size."""
    rng = np.random.default_rng(2000 + seed)
    spec = _random_ruleset(rng)
    book = _book(rng, scale=150.0)
    ks = np.array([0.5, 1.0, 1.5])
    idx = stationary_bootstrap(book.n_days, 50, 40, block_len=3, rng=rng)
    kw = dict(max_eval_days=25, funded_days=15, record_multipliers=[0, 2])
    whole = simulate(spec, book, idx, ks, **kw)
    chunked = simulate(spec, book, idx, ks, chunk_lanes=21, **kw)  # 7 paths per block
    for f in whole.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(chunked, f), getattr(whole, f), err_msg=f)
