"""
run_backtest.py - Entry point for VELOX backtests.

Wires all components together and runs the engine.
This is the file you run to execute a backtest.

Usage:
    python run_backtest.py
"""

import logging

from core.engine import Engine
from core.portfolio import Portfolio
from core.execution import SimulatedExecutionHandler
from core.risk import RiskManager
from data.historical import HistoricalDataHandler

# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt= "%H:%M:%S",
)

# ─── Configuration ────────────────────────────────────────────────────────────

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "PEP", "CVX"]
START_DATE  = "2020-01-01"
END_DATE    = "2023-12-31"
CAPITAL     = 100_000.0

# ─── Wire up components ───────────────────────────────────────────────────────

data = HistoricalDataHandler(
    symbols    = SYMBOLS,
    start_date = START_DATE,
    end_date   = END_DATE,
)

portfolio = Portfolio(
    data_handler    = data,
    initial_capital = CAPITAL,
)

execution = SimulatedExecutionHandler(
    data_handler   = data,
    slippage_pct   = 0.001,
    commission_pct = 0.001,
)

risk = RiskManager(
    portfolio          = portfolio,
    max_position_pct   = 0.20,
    max_open_positions = 10,
    max_drawdown_pct   = 0.20,
)

# ─── Run ──────────────────────────────────────────────────────────────────────

from signals.momentum import MomentumStrategy
from signals.pairs import KalmanPairsStrategy

momentum = MomentumStrategy(
    data_handler = data,
    symbols      = SYMBOLS,
    lookback     = 20,
    threshold    = 0.02,
)

pairs = KalmanPairsStrategy(
    data_handler   = data,
    pair           = ("PEP", "CVX"),
    z_score_entry  = 2.0,
    z_score_exit   = 0.5,
)

engine = Engine(
    data_handler      = data,
    strategies        = [pairs],
    portfolio         = portfolio,
    execution_handler = execution,
)

if __name__ == "__main__":
    logging.info("Starting VELOX backtest...")
    engine.run()

    print("\n── Results ──────────────────────────────")
    for k, v in portfolio.metrics.items():
        print(f"  {k:<22}: {v}")
    print("─────────────────────────────────────────")