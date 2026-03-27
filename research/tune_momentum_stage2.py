"""
Focused return-oriented search for momentum enhancements.

Stage A: signal quality (lookback/thresholds/holds + trend filter)
Stage B: sizing quality (vol targeting) around top Stage A configs
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
    trend_filter_ma: int
    vol_target: float
    vol_lookback: int
    max_vol_scale: float


def run_once(cfg: Config, symbols, start_date, end_date, capital, slippage_pct, commission_pct):
    data = HistoricalDataHandler(symbols=symbols, start_date=start_date, end_date=end_date)
    portfolio = Portfolio(data_handler=data, initial_capital=capital)
    execution = SimulatedExecutionHandler(
        data_handler=data, slippage_pct=slippage_pct, commission_pct=commission_pct
    )
    risk = RiskManager(portfolio=portfolio)
    regime = RegimeDetector(data_handler=data, symbol=symbols[0], lookback=252, retrain_every=126)
    strategy = MomentumStrategy(
        data_handler=data,
        symbols=symbols,
        lookback=cfg.lookback,
        enter_threshold=cfg.enter,
        exit_threshold=cfg.exit,
        min_hold_bars=cfg.min_hold,
        rebalance_every=cfg.rebalance,
        trend_filter_ma=cfg.trend_filter_ma,
        vol_target=cfg.vol_target,
        vol_lookback=cfg.vol_lookback,
        max_vol_scale=cfg.max_vol_scale,
        regime_detector=regime,
    )
    engine = Engine(
        data_handler=data,
        strategies=[regime, strategy],
        portfolio=portfolio,
        execution_handler=execution,
        risk_manager=risk,
    )
    engine.run()
    m = portfolio.metrics
    return {
        "total_return_pct": float(m.get("total_return_pct", 0.0)),
        "sharpe": float(m.get("sharpe_ratio", 0.0)),
        "max_drawdown_pct": float(m.get("max_drawdown_pct", 0.0)),
        "trades": int(m.get("total_trades", 0)),
        "final_equity": float(m.get("final_equity", capital)),
    }


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s | %(message)s")

    symbols = ["AAPL", "MSFT", "GOOGL", "PEP", "CVX"]
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    capital = 100_000.0
    slippage_pct = 0.001
    commission_pct = 0.001

    # Stage A: signal-quality search (no vol targeting)
    stage_a_grid = [
        Config(*vals, vol_target=0.0, vol_lookback=20, max_vol_scale=1.0)
        for vals in itertools.product(
            [40, 60],           # lookback
            [0.04, 0.05],       # enter
            [-0.02],            # exit
            [5, 10],            # min_hold
            [5, 10],            # rebalance
            [0, 200],           # trend filter MA (0 = off)
        )
    ]

    stage_a_rows = []
    for i, cfg in enumerate(stage_a_grid, start=1):
        r = run_once(cfg, symbols, start_date, end_date, capital, slippage_pct, commission_pct)
        row = cfg.__dict__.copy()
        row.update(r)
        stage_a_rows.append(row)
        print(
            f"[A {i:03d}/{len(stage_a_grid)}] lb={cfg.lookback} en={cfg.enter:.2f} ex={cfg.exit:.2f} "
            f"hold={cfg.min_hold} reb={cfg.rebalance} ma={cfg.trend_filter_ma} -> "
            f"ret={r['total_return_pct']:.2f}% sharpe={r['sharpe']:.3f}"
        )

    a_df = pd.DataFrame(stage_a_rows).sort_values(
        by=["total_return_pct", "sharpe"], ascending=[False, False]
    )
    top_a = a_df.head(4).copy()

    # Stage B: vol sizing around Stage A winners
    stage_b_rows = []
    vol_targets = [0.10, 0.12, 0.15]
    vol_lookbacks = [20, 40]
    max_scales = [1.25, 1.5]

    total_b = len(top_a) * len(vol_targets) * len(vol_lookbacks) * len(max_scales)
    b_i = 0
    for _, base in top_a.iterrows():
        for vt, vl, ms in itertools.product(vol_targets, vol_lookbacks, max_scales):
            b_i += 1
            cfg = Config(
                lookback=int(base["lookback"]),
                enter=float(base["enter"]),
                exit=float(base["exit"]),
                min_hold=int(base["min_hold"]),
                rebalance=int(base["rebalance"]),
                trend_filter_ma=int(base["trend_filter_ma"]),
                vol_target=vt,
                vol_lookback=vl,
                max_vol_scale=ms,
            )
            r = run_once(cfg, symbols, start_date, end_date, capital, slippage_pct, commission_pct)
            row = cfg.__dict__.copy()
            row.update(r)
            stage_b_rows.append(row)
            print(
                f"[B {b_i:03d}/{total_b}] vt={vt:.2f} vl={vl} ms={ms:.2f} "
                f"-> ret={r['total_return_pct']:.2f}% sharpe={r['sharpe']:.3f}"
            )

    b_df = pd.DataFrame(stage_b_rows).sort_values(
        by=["total_return_pct", "sharpe"], ascending=[False, False]
    )

    print("\n=== Stage A top 10 (signal-quality) ===")
    print(
        a_df[
            [
                "lookback", "enter", "exit", "min_hold", "rebalance", "trend_filter_ma",
                "total_return_pct", "sharpe", "max_drawdown_pct", "trades",
            ]
        ].head(10).to_string(index=False)
    )

    print("\n=== Stage B top 10 (with vol-target sizing) ===")
    print(
        b_df[
            [
                "lookback", "enter", "exit", "min_hold", "rebalance", "trend_filter_ma",
                "vol_target", "vol_lookback", "max_vol_scale",
                "total_return_pct", "sharpe", "max_drawdown_pct", "trades",
            ]
        ].head(10).to_string(index=False)
    )

    a_out = "research/output/momentum_stage2_signal_search.csv"
    b_out = "research/output/momentum_stage2_sizing_search.csv"
    a_df.to_csv(a_out, index=False)
    b_df.to_csv(b_out, index=False)
    print(f"\nSaved: {a_out}")
    print(f"Saved: {b_out}")


if __name__ == "__main__":
    main()

