"""Monte Carlo pipeline: statistics, EV maths, determinism, convergence, report."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from velox.app.cli import app
from velox.app.simulation import SimulationRequest, run_simulation
from velox.domain.daybook import DayBook, build_daybook
from velox.domain.sizing import IntratradeModel
from velox.ingest import load_trades
from velox.presets import load_preset
from velox.report.html import render_html
from velox.result import SimulationResult
from velox.sim.engine import LaneResults, simulate
from velox.sim.optimizer import multiplier_stats, wilson
from velox.sim.resample import stationary_bootstrap

FIX = Path(__file__).parent.parent / "fixtures"
JOURNAL = FIX / "futures_journal.csv"


@pytest.fixture(scope="module")
def journal():
    return load_trades(JOURNAL, timezone="America/Chicago")


# ─── statistics ──────────────────────────────────────────────────────────────


def test_wilson_interval_known_values():
    w = wilson(81, 100)  # textbook example
    assert (w.p, round(w.lo, 4), round(w.hi, 4)) == (0.81, 0.7222, 0.8749)
    z = wilson(0, 50)
    assert z.lo == 0 and 0.05 < z.hi < 0.08  # non-degenerate at 0, unlike Wald
    assert wilson(0, 0).hi == 1.0


def _lanes(outcome, n_payouts, trader, fee_paths=4):
    o = np.array([outcome])
    z = np.zeros_like(o)
    return LaneResults(
        multipliers=np.array([1.0]), outcome=o, fail_rule=z - 1, phase_reached=z,
        days_used=z + 5, consistency_delay_days=z, halted_days=z,
        funded_outcome=np.where(o == 0, 1, -1), funded_fail_rule=z - 1, funded_days_used=z,
        n_payouts=np.array([n_payouts]), gross_payouts=np.array([trader]) / 0.9,
        trader_payouts=np.array([trader], dtype=float), eval_balances=None)


def test_ev_and_breakeven_formulas():
    # 4 paths: 2 pass (one paid 900 to the trader, one never paid), 2 fail
    res = _lanes([0, 0, 2, 2], [1, 0, 0, 0], [900.0, 0.0, 0.0, 0.0])
    g = multiplier_stats(res, 0, fee=100.0, activation_fee=50.0, refund=False)
    # EV = E[net payouts] - fee - P(pass) * activation = 225 - 100 - 0.5 * 50
    assert g.ev == pytest.approx(100.0)
    assert g.breakeven_funded_payout == pytest.approx(100 / 0.5 + 50)
    assert g.p_pass.p == 0.5 and g.p_payout.p == 0.25
    # FTMO-style refund adds the fee back on paths that received a payout
    g2 = multiplier_stats(res, 0, fee=100.0, activation_fee=0.0, refund=True)
    assert g2.mean_net_payout == pytest.approx((900 + 100) / 4)


# ─── pipeline ────────────────────────────────────────────────────────────────


SMALL = dict(challenge_fee=49.0, n_paths=400, k_step=0.25, max_eval_days=40, funded_days=30,
             haircuts=[0.25, 0.5], seed=3)


@pytest.fixture(scope="module")
def result(journal):
    return run_simulation(journal, load_preset("topstep-50k"), SimulationRequest(**SMALL))


def test_result_shape_and_invariants(result: SimulationResult):
    ks = [g.k for g in result.grid]
    assert 1.0 in ks and ks == sorted(ks)
    for g in result.grid:
        total = g.p_pass.p + g.p_fail_daily_loss.p + g.p_fail_max_loss.p + g.p_timeout.p
        assert total == pytest.approx(1.0)
        assert g.p_pass.lo <= g.p_pass.p <= g.p_pass.hi
        assert g.ev_lo <= g.ev <= g.ev_hi
        assert g.p_payout.p <= g.p_pass.p  # only funded paths can be paid
    rec = result.recommendation
    assert rec.ev == max(g.ev for g in result.grid)
    assert [s.haircut for s in result.stress] == [0.25, 0.5]
    for s in result.stress:  # stress includes current and recommended sizes
        assert {rec.k, rec.k_current} <= {p.k for p in s.grid}
    # removing edge can only hurt at a fixed size (same paths, same rules)
    evs = [result.recommendation.ev] + [s.at_base_recommendation.ev for s in result.stress]
    assert evs == sorted(evs, reverse=True)
    assert len(result.fans) in (1, 2) and len(result.fans[0].quantiles["p50"]) == 41


def test_same_seed_gives_identical_json(journal, result):
    again = run_simulation(journal, load_preset("topstep-50k"), SimulationRequest(**SMALL))
    assert again.model_dump_json() == result.model_dump_json()
    other = run_simulation(journal, load_preset("topstep-50k"),
                           SimulationRequest(**{**SMALL, "seed": 4}))
    assert other.model_dump_json() != result.model_dump_json()


def test_json_round_trip(result):
    assert SimulationResult.model_validate_json(result.model_dump_json()) == result


def test_pass_probability_converges_across_seeds(journal):
    """At 10,000 paths, the pass probability at 1.0x is stable to +-1pp across 5 seeds."""
    spec = load_preset("topstep-50k")
    book = build_daybook(journal, spec.calendar)
    ps = []
    for seed in range(5):
        idx = stationary_bootstrap(book.n_days, 10_000, 60, rng=np.random.default_rng(seed))
        r = simulate(spec, book, idx, np.array([1.0]), max_eval_days=60)
        ps.append(float((r.outcome[0] == 0).mean()))
    assert max(ps) - min(ps) <= 0.02  # every pair within 2pp => each within 1pp of the mean
    assert all(abs(p - np.mean(ps)) <= 0.01 for p in ps)


def test_haircut_reduces_mean_edge_exactly():
    book = DayBook(np.arange(2).astype("datetime64[D]"), np.array([0, 2, 3]),
                   np.array([1.0, 2.0, 1.0]), np.array([100.0, -20.0, 40.0]),
                   np.array([0.0, 20.0, 0.0]), IntratradeModel.MAE)
    mu = np.mean(book.pnl_per_unit * book.units)  # (100 - 40 + 40) / 3
    h = book.haircut(0.5)
    assert np.mean(h.pnl_per_unit * h.units) == pytest.approx(0.5 * mu)
    assert np.all(h.adverse_per_unit >= -h.pnl_per_unit - 1e-12)


# ─── report ──────────────────────────────────────────────────────────────────


def test_html_report_is_self_contained(result):
    html = render_html(result)
    assert "<script src" not in html and "<link" not in html
    assert not re.search(r"""(src|href)=["']https?://""", html)
    assert "renderVelox" in html
    # the embedded payload parses back to the same result
    marker = '<script type="application/json" id="velox-data">'
    payload = html.split(marker)[1].split("</script>")[0]
    assert SimulationResult.model_validate(json.loads(payload.replace("<\\/", "</"))) == result


def test_cli_simulate_writes_json_and_html(tmp_path):
    out, js = tmp_path / "r.html", tmp_path / "r.json"
    res = CliRunner().invoke(app, [
        "simulate", str(JOURNAL), "-p", "topstep-50k", "--tz", "America/Chicago",
        "--fee", "49", "--paths", "300", "--k-step", "0.5", "--funded-days", "20",
        "--payout-cap", "2000", "--max-eval-days", "30", "--out", str(out), "--json", str(js)])
    assert res.exit_code == 0, res.output
    r = SimulationResult.model_validate_json(js.read_text("utf-8"))
    assert r.config.payout_cap == 2000 and r.config.funded_days == 20
    assert out.read_text("utf-8").startswith("<!doctype html>")
    assert "recommended" in res.output
