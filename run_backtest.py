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
from research.tearsheet import Tearsheet
from signals.regime import RegimeDetector
from research.visualizer import Visualizer

# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt= "%H:%M:%S",
)

# ─── Configuration ────────────────────────────────────────────────────────────

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "JPM", "PEP", "CVX"]
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
from signals.ml_signal import MLSignalStrategy

regime = RegimeDetector(
    data_handler   = data,
    symbol         = "AAPL",
    n_states       = 3,
    lookback       = 252,
    retrain_every  = 63,
)

momentum = MomentumStrategy(
    data_handler = data,
    symbols      = SYMBOLS,
    lookback     = 20,
    enter_threshold = 0.03,
    exit_threshold  = -0.01,
    min_hold_bars   = 5,
    rebalance_every = 5,
    regime_detector = regime,
)

ml = MLSignalStrategy(
    data_handler = data,
    symbols      = SYMBOLS,
    lookback     = 252,
    retrain_every= 21,
    threshold    = 0.6,
    regime_detector = regime,
)

engine = Engine(
    data_handler      = data,
    strategies        = [regime, ml],
    portfolio         = portfolio,
    execution_handler = execution,
    risk_manager      = risk,
)

if __name__ == "__main__":
    logging.info("Starting VELOX backtest...")
    engine.run()

    tearsheet = Tearsheet(portfolio, output_dir="research/output")
    full_metrics = tearsheet.generate(strategy_name="ml_signal")

    viz = Visualizer(portfolio, regime_detector=regime, data_handler=data)
    viz.generate(symbol="AAPL", name="velox")

    print("\n── Results ──────────────────────────────")
    for k, v in portfolio.metrics.items():
        print(f"  {k:<22}: {v}")
    print("─────────────────────────────────────────")