# VELOX

**Prop-firm challenge survival and position-sizing simulator.**

Prop-firm evaluations (FTMO, Topstep, ...) are usually failed on position size, not on edge:
a normal losing streak at the wrong size breaches a daily-loss or trailing-drawdown rule.
VELOX takes your own trade log and answers three questions:

1. At my current size, what is my probability of passing, and how do I most likely fail?
2. What size multiplier maximises expected value (funded payouts minus challenge fees)?
3. What does that look like? (a shareable, offline HTML tearsheet)

> Status: **Phase 2 of 3**. Ingestion, rule engine, presets, `replay`, Monte Carlo, the
> sizing optimizer and the HTML/JSON report all work. Phase 3 is the local web simulator.

## Quickstart

```bash
uv sync
uv run velox presets
uv run velox inspect  tests/fixtures/futures_journal.csv -p topstep-50k --tz America/Chicago
uv run velox replay   tests/fixtures/futures_journal.csv -p topstep-50k --tz America/Chicago
uv run velox simulate tests/fixtures/futures_journal.csv -p topstep-50k --tz America/Chicago \
    --fee 49 --out out/report.html --json out/result.json
```

On OneDrive/Dropbox folders, keep the virtualenv outside the synced folder
(`UV_PROJECT_ENVIRONMENT=<path outside the sync folder>`). Sync clients lock files in `.venv`.

## How a simulation works

1. **Ingest:** the log is normalised to UTC trades.
2. **Group into days:** trades are grouped into the firm's trading days. FTMO resets at midnight
   Prague time; Topstep rolls over at 17:00 Chicago time.
3. **Resample:** each path replays whole historical trading days. The default is a stationary
   block bootstrap (`--block-len` days on average), which keeps losing streaks together.
   `--resampler regime` uses an HMM fitted to your daily P&L (needs >= 60 days and
   `velox[regime]`).
4. **Size grid:** every multiplier on the grid (`--k-min/--k-max/--k-step`, default 0.25-2.0 in
   steps of 0.05) replays the *same* paths (common random numbers). Trade units are rescaled per
   trade (`--unit-step`, `--max-units` for whole contracts and caps), then each path runs
   through the evaluation phases (capped at `--max-eval-days`) and, if it passes, the funded
   stage for `--funded-days` (default 60). The funded stage uses the firm's funded drawdown
   rules and payout policy.
5. **Economics:**
   `EV(k) = E[net funded payouts] - challenge fee - P(pass) x activation fee`.
   - Net payouts are the profit split times gross payouts, plus the fee refund where the firm
     refunds it.
   - Paths that fail are worth 0.
   - Payouts already taken before a funded breach are kept.
   - Profit left unwithdrawn at the horizon is not counted.

   `breakeven funded payout = fee / P(pass) + activation fee` is reported at every size as a
   model-free check.
6. **Uncertainty:**
   - every probability carries a 95% Wilson interval and EV a 95% normal interval;
   - 25% and 50% edge-haircut stress tests re-run the same paths with that share of your
     average edge removed.

Everything lands in one Pydantic `SimulationResult` (schema 1.0). `--json` writes it; the HTML
tearsheet is rendered from it by `report/assets/renderer.js`. Same inputs and seed produce
byte-identical JSON.

The step-shaped curves are real, not noise. A finite history has a few worst days. Once the
size multiplier pushes one of them past a daily-loss limit, every path containing that day
fails, so pass probability drops in steps. Small samples exaggerate this: see the data-quality
warnings.

### Trade log formats

| `--format` | file | notes |
|---|---|---|
| `generic` (default) | CSV with `open_time, close_time, pnl` [+ `symbol, units, risk, mae`] | `pnl` net of costs; `risk`/`mae` positive amounts |
| `mt5` | MT5 *Positions* history export | UTF-16/tab (desktop) or UTF-8/comma; commission + swap added; totals row dropped |
| `tradovate` | Tradovate *Performance* report | partial fills merged; P&L is gross, so use `--commission-per-unit` |

Naive timestamps need `--tz`, because broker server time is rarely UTC.

### Intra-trade (floating) loss model: read this

Firms check equity *while trades are open*, but most logs only record realised P&L.
`--intratrade` chooses the assumption:

| model | assumption | bias |
|---|---|---|
| `realized` | equity dips only to the realised loss | optimistic |
| `mae` | equity dips to the recorded MAE | best available |
| `stop_bound` | every trade dips to its full planned risk | pessimistic |
| `auto` (default) | `mae` if present, else `realized` | - |

## Rule presets

The presets in `src/velox/presets/*.yaml` are versioned data. Each one records `source_urls`,
`verified_at` and the `needs_verification` items it still has open, and covers the evaluation
phases plus the funded stage. Bundled presets:

- `ftmo-2step`
- `ftmo-1step` (Consistency rule)
- `ftmo-1step-bestday` (legacy Best Day rule)
- `topstep-50k` (Combine + Express Funded Account)

A custom firm is just another YAML file (`--preset path/to/firm.yaml`).

Supported rule types:

- **Floors:** static max loss, EOD trailing max loss and intraday trailing max loss (each with
  a lock level and an optional post-payout lock), and daily loss limit (fail or halt-for-day).
- **Pass conditions:** profit target, minimum trading days, best-day consistency
  (total-profit or positive-days denominator).
- **Funded stage:** profit split, payout eligibility (days between payouts, winning days over a
  threshold), payout share/cap, fee refund, activation fee.

VELOX is not affiliated with any prop firm. Firm rules change often, so confirm them on your
own account. Nothing here is financial advice.

## Layout

```
src/velox/
  domain/    trades, calendar, sizing, daybook, stats, rules/{spec,reference}   (pure)
  ingest/    generic CSV, MT5, Tradovate, data-quality report
  presets/   versioned firm rule YAMLs
  sim/       resamplers, regime HMM, vectorised engine, optimizer (EV, Wilson CIs)
  report/    SimulationResult schema, build (orchestration), HTML tearsheet + renderer.js
  app/       CLI (Phase 3: FastAPI + local web UI)
tests/       golden rule scenarios, engine-vs-reference parity, adapters, pipeline
```

The rule maths lives once, in `domain/rules/spec.py`, written with NumPy operations. The plain
Python reference engine (`domain/rules/reference.py`) and the vectorised engine
(`sim/engine.py`) both call it. `tests/unit/test_engine_parity.py` checks them against each
other outcome-for-outcome on more than 2,000 randomised multi-stage paths.

## Development

```bash
uv run pytest
uv run ruff check .
```

The pre-pivot equity backtester is archived on branch `legacy/backtester` and tag
`v0-legacy-backtester`.
