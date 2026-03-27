"""
Grid-search momentum parameters against buy-and-hold baseline.

Runs cost-aware backtests and reports the best parameter sets.
"""

import itertools
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.engine import Engine
from core.execution import SimulatedExecutionHandler
from core.portfolio import Portfolio
from core.risk import RiskManager
from data.historical import HistoricalDataHandler
from signals.momentum import MomentumStrategy
from signals.regime import RegimeDetector


@dataclass
class Config:
    lookback: int
    enter: float
    exit: float
    min_hold: int
    rebalance: int


def buyhold_return_pct(data: HistoricalDataHandler, capital: float) -> float:
    series = {}
    for sym, df in data.all_bars.items():
        if "close" not in df.columns or df.empty:
            continue
        close = df["close"].astype(float)
        series[sym] = close / float(close.iloc[0])
    if not series:
        return 0.0

    combined = pd.DataFrame(series).mean(axis=1)
    final_equity = float(combined.iloc[-1] * capital)
    return (final_equity - capital) / capital * 100.0


def run_once(
    symbols: list[str],
    start_date: str,
    end_date: str,
    capital: float,
    slippage_pct: float,
    commission_pct: float,
    cfg: Config,
) -> dict:
    data = HistoricalDataHandler(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )
    portfolio = Portfolio(data_handler=data, initial_capital=capital)
    execution = SimulatedExecutionHandler(
        data_handler=data,
        slippage_pct=slippage_pct,
        commission_pct=commission_pct,
    )
    risk = RiskManager(portfolio=portfolio)
    regime = RegimeDetector(
        data_handler=data,
        symbol=symbols[0],
        lookback=252,
        retrain_every=126,
    )
    momentum = MomentumStrategy(
        data_handler=data,
        symbols=symbols,
        lookback=cfg.lookback,
        enter_threshold=cfg.enter,
        exit_threshold=cfg.exit,
        min_hold_bars=cfg.min_hold,
        rebalance_every=cfg.rebalance,
        regime_detector=regime,
    )
    engine = Engine(
        data_handler=data,
        strategies=[regime, momentum],
        portfolio=portfolio,
        execution_handler=execution,
        risk_manager=risk,
    )
    engine.run()
    m = portfolio.metrics

    return {
        "lookback": cfg.lookback,
        "enter": cfg.enter,
        "exit": cfg.exit,
        "min_hold": cfg.min_hold,
        "rebalance": cfg.rebalance,
        "total_return_pct": m.get("total_return_pct", 0.0),
        "ann_return_pct": m.get("ann_return_pct", 0.0),
        "sharpe": m.get("sharpe_ratio", 0.0),
        "max_drawdown_pct": m.get("max_drawdown_pct", 0.0),
        "trades": m.get("total_trades", 0),
        "final_equity": m.get("final_equity", capital),
        "buyhold_return_pct": buyhold_return_pct(data, capital),
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    symbols = ["AAPL", "MSFT", "GOOGL", "PEP", "CVX"]
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    capital = 100_000.0
    slippage_pct = 0.001
    commission_pct = 0.001

    grid = [
        Config(*vals)
        for vals in itertools.product(
            [20, 40],                # lookback
            [0.02, 0.03, 0.04],      # enter threshold
            [-0.02, -0.01],          # exit threshold
            [5, 10],                 # min hold bars
            [5],                     # rebalance frequency
        )
    ]

    rows = []
    for i, cfg in enumerate(grid, start=1):
        result = run_once(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            capital=capital,
            slippage_pct=slippage_pct,
            commission_pct=commission_pct,
            cfg=cfg,
        )
        result["outperformance_pct"] = round(
            result["total_return_pct"] - result["buyhold_return_pct"], 2
        )
        rows.append(result)
        print(
            f"[{i:02d}/{len(grid)}] lb={cfg.lookback} en={cfg.enter:.2f} ex={cfg.exit:.2f} "
            f"hold={cfg.min_hold} -> ret={result['total_return_pct']:.2f}% "
            f"sharpe={result['sharpe']:.3f} trades={result['trades']}"
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["outperformance_pct", "sharpe", "total_return_pct"],
        ascending=[False, False, False],
    )

    print("\n=== Top 10 configs by outperformance vs buy-and-hold ===")
    print(
        df[
            [
                "lookback",
                "enter",
                "exit",
                "min_hold",
                "rebalance",
                "total_return_pct",
                "buyhold_return_pct",
                "outperformance_pct",
                "sharpe",
                "max_drawdown_pct",
                "trades",
                "final_equity",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    out_path = "research/output/momentum_tuning_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved full results: {out_path}")


if __name__ == "__main__":
    main()

