"""`velox` command-line interface.

Commands:
    velox presets                                 list bundled rule presets
    velox inspect  LOG --preset P --tz TZ         parse a trade log, report data quality
    velox replay   LOG --preset P --tz TZ         run the actual history through the rules
    velox simulate LOG --preset P --tz TZ --fee F Monte Carlo sizes, write the tearsheet
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from velox.domain.daybook import build_daybook
from velox.domain.outcomes import Outcome
from velox.domain.rules.reference import PathResult, run_path
from velox.domain.rules.spec import RulesetSpec
from velox.domain.sizing import IntratradeModel, SizingModel
from velox.domain.stats import summarize
from velox.domain.trades import TradeLog, TradeLogError
from velox.ingest import load_trades
from velox.ingest.quality import assess
from velox.presets import list_presets, load_preset

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Prop-firm challenge survival & position-sizing simulator.")

LogArg = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Trade log CSV")]
PresetOpt = Annotated[str, typer.Option("--preset", "-p", help="Preset id or YAML path")]
TzOpt = Annotated[str | None, typer.Option("--tz", help="Source timezone of naive timestamps")]
FmtOpt = Annotated[str, typer.Option("--format", "-f", help="generic | mt5 | tradovate")]
CommOpt = Annotated[float, typer.Option(
    "--commission-per-unit", help="Round-turn commission per contract to deduct. Tradovate "
    "only (it reports gross P&L); other formats are already net and reject it")]
BalanceOpt = Annotated[float | None, typer.Option("--balance", help="Override account size")]
IntraOpt = Annotated[IntratradeModel, typer.Option("--intratrade", help="Floating-loss model")]


def _load(log: Path, fmt: str, tz: str | None, commission: float = 0.0) -> TradeLog:
    try:
        return load_trades(log, fmt, tz, commission_per_unit=commission)  # type: ignore[arg-type]
    except (TradeLogError, ValueError) as e:
        typer.secho(f"error: {e}", fg="red", err=True)
        raise typer.Exit(2) from e


def _ruleset(preset: str, balance: float | None) -> RulesetSpec:
    try:
        spec = load_preset(preset)
    except KeyError as e:
        typer.secho(f"error: {e.args[0]}", fg="red", err=True)
        raise typer.Exit(2) from e
    return spec.with_balance(balance) if balance else spec


@app.command()
def presets() -> None:
    """List bundled rule presets."""
    for s in list_presets():
        typer.echo(f"{s.id:<14} {s.name}  (balance {s.initial_balance:,.0f}, "
                   f"verified {s.verified_at}, {len(s.needs_verification)} open questions)")


@app.command()
def inspect(log: LogArg, preset: PresetOpt, tz: TzOpt = None, fmt: FmtOpt = "generic",
            intratrade: IntraOpt = IntratradeModel.AUTO, commission: CommOpt = 0.0) -> None:
    """Parse a trade log and report data quality under a preset's trading-day calendar."""
    trades = _load(log, fmt, tz, commission)
    spec = _ruleset(preset, None)
    q = assess(trades, spec.calendar, intratrade)
    book = build_daybook(trades, spec.calendar, intratrade)
    t, d = summarize(trades.pnl), summarize(book.daily_pnl())
    typer.echo(f"{q.n_trades} trades over {q.n_trading_days} trading days "
               f"({q.first_trade:%Y-%m-%d} to {q.last_trade:%Y-%m-%d}, {spec.calendar.timezone} "
               f"days rolling at {spec.calendar.rollover:%H:%M})")
    typer.echo(f"trades: total {t.total:,.2f}  win rate {t.win_rate:.1%}  "
               f"PF {_pf(t.profit_factor)}  avg win {t.avg_win:,.2f}  avg loss {t.avg_loss:,.2f}  "
               f"worst losing streak {t.longest_losing_streak}")
    typer.echo(f"days:   mean {d.mean:,.2f}  best {d.best:,.2f}  worst {d.worst:,.2f}  "
               f"VaR95 {d.var_95:,.2f}")
    typer.echo(f"intra-trade model: {q.intratrade_model}")
    for w in q.warnings:
        typer.secho(f"warning: {w}", fg="yellow")


def _pf(v: float | None) -> str:
    return "n/a (no losses)" if v is None else f"{v:.2f}"


def _describe(r: PathResult, spec: RulesetSpec) -> str:
    phase = spec.phases[min(r.phase_reached, len(spec.phases) - 1)].name
    match r.outcome:
        case Outcome.PASSED:
            return f"PASSED all phases in {r.days_used} trading days"
        case Outcome.FAIL_DAILY_LOSS | Outcome.FAIL_MAX_LOSS:
            return f"FAILED in {phase}: {r.fail_rule} on trading day {r.days_used}"
        case _:
            return f"NOT FINISHED: history ran out during {phase} after {r.days_used} days"


@app.command()
def replay(log: LogArg, preset: PresetOpt, tz: TzOpt = None, fmt: FmtOpt = "generic",
           balance: BalanceOpt = None, intratrade: IntraOpt = IntratradeModel.AUTO,
           k: Annotated[float, typer.Option("--k", help="Size multiplier")] = 1.0,
           unit_step: Annotated[float | None, typer.Option(help="Round units to step")] = None,
           max_units: Annotated[float | None, typer.Option(help="Cap units per trade")] = None,
           attempts: Annotated[bool, typer.Option(
               help="After a fail/pass, start a new attempt on the next day")] = True,
           commission: CommOpt = 0.0) -> None:
    """Replay the actual trade history, in order, through the challenge rules."""
    trades = _load(log, fmt, tz, commission)
    spec = _ruleset(preset, balance)
    book = build_daybook(trades, spec.calendar, intratrade)
    pnl, adv, units = book.scaled(k, SizingModel(unit_step=unit_step, max_units=max_units))
    day_iter = iter([(pnl[book.day_slice(d)], adv[book.day_slice(d)], units[book.day_slice(d)])
                     for d in range(book.n_days)])

    typer.echo(f"{spec.name} @ {spec.initial_balance:,.0f}, size x{k:g}, "
               f"intra-trade model: {book.intratrade}")
    n, day0 = 0, 0
    while True:
        r = run_path(spec, day_iter)
        if r.days_used == 0:
            break
        n += 1
        start = str(book.days[day0])
        typer.echo(f"  attempt {n} (from {start}): {_describe(r, spec)}")
        day0 += r.days_used
        if r.outcome is Outcome.INCOMPLETE or not attempts:
            break
    for note in spec.needs_verification:
        typer.secho(f"rule caveat: {note}", fg="yellow")


@app.command()
def simulate(
    log: LogArg, preset: PresetOpt,
    fee: Annotated[float, typer.Option("--fee", help="Challenge fee you pay per attempt")],
    tz: TzOpt = None, fmt: FmtOpt = "generic", balance: BalanceOpt = None,
    intratrade: IntraOpt = IntratradeModel.AUTO, commission: CommOpt = 0.0,
    activation_fee: Annotated[float | None, typer.Option(
        help="Funded activation fee (default: preset value)")] = None,
    paths: Annotated[int, typer.Option("--paths", help="Simulated attempts per size")] = 10_000,
    seed: Annotated[int, typer.Option(help="Random seed (same seed -> identical JSON)")] = 7,
    k_min: Annotated[float, typer.Option(help="Smallest size multiplier")] = 0.25,
    k_max: Annotated[float, typer.Option(help="Largest size multiplier")] = 2.0,
    k_step: Annotated[float, typer.Option(help="Multiplier grid step")] = 0.05,
    resampler: Annotated[str, typer.Option(help="stationary | regime")] = "stationary",
    block_len: Annotated[float, typer.Option(help="Mean bootstrap block length (days)")] = 4.0,
    max_eval_days: Annotated[int, typer.Option(help="Evaluation cap (trading days)")] = 60,
    funded_days: Annotated[int, typer.Option(help="Funded horizon (trading days)")] = 60,
    payout_cap: Annotated[float | None, typer.Option(help="Max gross amount per payout")] = None,
    haircuts: Annotated[str, typer.Option(help="Comma-separated edge haircuts")] = "0.25,0.5",
    unit_step: Annotated[float | None, typer.Option(help="Round units to step")] = None,
    max_units: Annotated[float | None, typer.Option(help="Cap units per trade")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="HTML tearsheet path")] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="SimulationResult JSON path")]
    = None,
) -> None:
    """Monte Carlo the challenge across position sizes and write the tearsheet."""
    import time

    from velox.app.simulation import SimulationRequest, run_simulation
    from velox.report.html import render_html

    trades = _load(log, fmt, tz, commission)
    spec = _ruleset(preset, balance)
    try:
        req = SimulationRequest(
            challenge_fee=fee, activation_fee=activation_fee, n_paths=paths, seed=seed,
            k_min=k_min, k_max=k_max, k_step=k_step, resampler=resampler, block_len=block_len,
            max_eval_days=max_eval_days, funded_days=funded_days, payout_cap=payout_cap,
            haircuts=[float(h) for h in haircuts.split(",") if h.strip()],
            unit_step=unit_step, max_units=max_units, intratrade=intratrade,
        )
    except ValueError as e:
        typer.secho(f"error: {e}", fg="red", err=True)
        raise typer.Exit(2) from e

    t0 = time.perf_counter()
    result = run_simulation(trades, spec, req)
    elapsed = time.perf_counter() - t0
    rec = result.recommendation
    by_k = {g.k: g for g in result.grid}
    cur, best = by_k[rec.k_current], by_k[rec.k]
    typer.echo(f"{spec.name}: {len(result.grid)} sizes x {paths:,} paths in {elapsed:.1f}s")
    typer.echo(f"  at your size x{cur.k:g}: pass {cur.p_pass.p:.1%} "
               f"[{cur.p_pass.lo:.1%}, {cur.p_pass.hi:.1%}]  "
               f"daily-loss {cur.p_fail_daily_loss.p:.1%}"
               f"  max-loss {cur.p_fail_max_loss.p:.1%}  delayed {cur.p_consistency_delay.p:.1%}"
               f"  EV {cur.ev:+,.0f}  break-even payout {_money(cur.breakeven_funded_payout)}")
    typer.echo(f"  recommended  x{best.k:g}: pass {best.p_pass.p:.1%} "
               f"[{best.p_pass.lo:.1%}, {best.p_pass.hi:.1%}]  EV {best.ev:+,.0f} "
               f"[{best.ev_lo:+,.0f}, {best.ev_hi:+,.0f}]")
    for s in result.stress:
        typer.echo(f"  {s.haircut:.0%} edge haircut: best x{s.recommended_k:g}; "
                   f"x{best.k:g} -> pass {s.at_base_recommendation.p_pass.p:.1%}, "
                   f"EV {s.at_base_recommendation.ev:+,.0f}")
    for w in result.warnings:
        typer.secho(f"warning: {w}", fg="yellow")
    if json_out:
        json_out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"wrote {json_out}")
    if out:
        out.write_text(render_html(result), encoding="utf-8")
        typer.echo(f"wrote {out}")


def _money(v: float | None) -> str:
    return "n/a" if v is None else f"{v:,.0f}"
