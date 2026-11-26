# VELOX

**Prop-firm challenge survival and position-sizing simulator.**

Prop-firm evaluations (FTMO, Topstep, ...) are usually failed on position size, not on edge:
a normal losing streak at the wrong size breaches a daily-loss or trailing-drawdown rule.
VELOX takes your own trade log and answers three questions:

1. At my current size, what is my probability of passing, and how do I most likely fail?
2. What size multiplier maximises expected value (payout odds minus challenge fees)?
3. What does that look like? (a shareable tearsheet)

> Status: **Phase 1 of 3**. The trade-log ingestion, firm calendars, rule engine,
> presets and historical `replay` work. Monte Carlo, the sizing optimizer and the HTML report
> are Phase 2. The local web simulator is Phase 3.

## Quickstart

```bash
uv sync
uv run velox presets
uv run velox inspect tests/fixtures/topstep_sample.csv -p topstep-50k --tz America/Chicago
uv run velox replay  tests/fixtures/topstep_sample.csv -p topstep-50k --tz America/Chicago
```

On OneDrive/Dropbox folders, keep the virtualenv outside the synced folder
(`UV_PROJECT_ENVIRONMENT=<path outside the sync folder>`). Sync clients lock files in `.venv`.

### Trade log format

Canonical CSV columns are `open_time, close_time, pnl`, plus the optional `symbol, units, risk, mae`.

- `pnl` is the net realised P&L (commission and swap included).
- `risk` is the planned loss at the stop, as a positive amount.
- `mae` is the max adverse excursion, as a positive amount.
- `--format mt5` reads a MetaTrader 5 *Positions* export (commission + swap added to profit).
- Naive timestamps need `--tz`, because broker server time is rarely UTC.

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
`verified_at` and the `needs_verification` items it still has open. Bundled:
`ftmo-2step`, `ftmo-1step` (Consistency rule), `ftmo-1step-bestday` (legacy), `topstep-50k`. A custom firm is just another YAML file
(`--preset path/to/firm.yaml`).

Supported rule types:

- static max loss
- EOD trailing max loss (with lock)
- intraday trailing max loss (with lock)
- daily loss limit (fail or halt-for-day)
- profit target
- minimum trading days
- best-day consistency (total-profit or positive-days denominator)
- multi-phase evaluations

VELOX is not affiliated with any prop firm. Firm rules change often, so confirm them on your
own account. Nothing here is financial advice.

## Layout

```
src/velox/
  domain/    trades, calendar, sizing, daybook, stats, rules/{spec,reference}   (pure)
  ingest/    generic column-mapped CSV, MT5, data-quality report
  presets/   versioned firm rule YAMLs
  sim/       regime HMM (Phase 2: bootstrap, vectorised engine, optimizer)
  report/    (Phase 2: SimulationResult schema + HTML tearsheet)
  app/       CLI (Phase 3: FastAPI + local web UI)
tests/       golden rule scenarios (hand-computed), unit tests
```

## Development

```bash
uv run pytest
uv run ruff check src tests
```

The pre-pivot equity backtester is archived on branch `legacy/backtester` and tag
`v0-legacy-backtester`.
