"""`velox` command-line interface.

Phase 1 commands:
    velox presets                         list bundled rule presets
    velox inspect LOG --preset P --tz TZ  parse a trade log and report data quality
    velox replay  LOG --preset P --tz TZ  run the actual history through the rules
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from velox.domain.daybook import build_daybook
from velox.domain.rules.reference import Outcome, PathResult, run_path
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
FmtOpt = Annotated[str, typer.Option("--format", "-f", help="generic | mt5")]
BalanceOpt = Annotated[float | None, typer.Option("--balance", help="Override account size")]
IntraOpt = Annotated[IntratradeModel, typer.Option("--intratrade", help="Floating-loss model")]


def _load(log: Path, fmt: str, tz: str | None) -> TradeLog:
    try:
        return load_trades(log, fmt, tz)  # type: ignore[arg-type]
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
            intratrade: IntraOpt = IntratradeModel.AUTO) -> None:
    """Parse a trade log and report data quality under a preset's trading-day calendar."""
    trades = _load(log, fmt, tz)
    spec = _ruleset(preset, None)
    q = assess(trades, spec.calendar, intratrade)
    book = build_daybook(trades, spec.calendar, intratrade)
    t, d = summarize(trades.pnl), summarize(book.daily_pnl())
    typer.echo(f"{q.n_trades} trades over {q.n_trading_days} trading days "
               f"({q.first_trade:%Y-%m-%d} to {q.last_trade:%Y-%m-%d}, {spec.calendar.timezone} "
               f"days rolling at {spec.calendar.rollover:%H:%M})")
    typer.echo(f"trades: total {t.total:,.2f}  win rate {t.win_rate:.1%}  "
               f"PF {t.profit_factor:.2f}  avg win {t.avg_win:,.2f}  avg loss {t.avg_loss:,.2f}  "
               f"worst losing streak {t.longest_losing_streak}")
    typer.echo(f"days:   mean {d.mean:,.2f}  best {d.best:,.2f}  worst {d.worst:,.2f}  "
               f"VaR95 {d.var_95:,.2f}")
    typer.echo(f"intra-trade model: {q.intratrade_model}")
    for w in q.warnings:
        typer.secho(f"warning: {w}", fg="yellow")


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
               help="After a fail/pass, start a new attempt on the next day")] = True) -> None:
    """Replay the actual trade history, in order, through the challenge rules."""
    trades = _load(log, fmt, tz)
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
