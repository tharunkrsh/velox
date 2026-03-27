"""
api/main.py - FastAPI backend for VELOX.

Exposes the backtest engine as a REST API.
The React dashboard calls these endpoints to run
backtests and retrieve results.

Run with:
    uvicorn api.main:app --reload --port 8000

Docs auto-generated at:
    http://localhost:8000/docs
"""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.engine import Engine
from core.portfolio import Portfolio
from core.execution import SimulatedExecutionHandler
from core.risk import RiskManager
from data.historical import HistoricalDataHandler
from signals.regime import RegimeDetector
from signals.momentum import MomentumStrategy
from signals.ml_signal import MLSignalStrategy
from signals.pairs import KalmanPairsStrategy

logger = logging.getLogger(__name__)

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "VELOX API",
    description = "Algorithmic trading research framework — REST API",
    version     = "1.0.0",
)

# Allow React dashboard to call the API from a different port
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─── Request / Response models ────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    symbols:        List[str]  = Field(default=["AAPL", "MSFT", "GOOGL", "PEP", "CVX"])
    start_date:     str        = Field(default="2020-01-01")
    end_date:       str        = Field(default="2023-12-31")
    initial_capital: float     = Field(default=100_000.0)
    strategy:       str        = Field(default="ml")        # "ml", "momentum", "ml+momentum"
    ml_threshold:   float      = Field(default=0.6)
    slippage_pct:   float      = Field(default=0.001)
    commission_pct: float      = Field(default=0.001)
    momentum_lookback: int     = Field(default=20)
    momentum_enter_threshold: float = Field(default=0.03)
    momentum_exit_threshold: float  = Field(default=-0.01)
    momentum_min_hold_bars: int     = Field(default=5)
    momentum_rebalance_every: int   = Field(default=5)


class EquityPoint(BaseModel):
    timestamp:    str
    total_equity: float
    cash:         float
    market_value: float
    buyhold_norm: float = 100.0


class RegimePoint(BaseModel):
    timestamp:    str
    regime:       str
    prob_bull:    float
    prob_sideways: float
    prob_bear:    float


class BacktestResponse(BaseModel):
    status:         str
    duration_secs:  float
    metrics:        dict
    equity_curve:   List[EquityPoint]
    regime_history: List[RegimePoint]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check if the API is alive."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/strategies")
def strategies():
    """List available strategies."""
    return {
        "strategies": [
            {
                "id":          "ml",
                "name":        "LightGBM ML Signal",
                "description": "Walk-forward gradient boosted classifier with RSI, momentum, and volatility features.",
            },
            {
                "id":          "momentum",
                "name":        "Time-Series Momentum",
                "description": "Jegadeesh & Titman (1993) momentum factor with regime filtering.",
            },
            {
                "id":          "ml+momentum",
                "name":        "ML + Momentum Combined",
                "description": "Both strategies running simultaneously with HMM regime gating.",
            },
        ]
    }


@app.post("/backtest", response_model=BacktestResponse)
def run_backtest(request: BacktestRequest):
    """
    Run a full backtest and return results.

    This is the main endpoint — the React dashboard calls this
    when the user clicks 'Run Backtest'.
    """
    start_time = datetime.utcnow()

    try:
        # ── Wire up components ────────────────────────────────────────────────
        if request.strategy in ("pairs", "ml+pairs"):
            for sym in ["PEP", "CVX"]:
                if sym not in request.symbols:
                    request.symbols.append(sym)

        data = HistoricalDataHandler(
            symbols    = request.symbols,
            start_date = request.start_date,
            end_date   = request.end_date,
        )

        portfolio = Portfolio(
            data_handler    = data,
            initial_capital = request.initial_capital,
        )

        execution = SimulatedExecutionHandler(
            data_handler   = data,
            slippage_pct   = request.slippage_pct,
            commission_pct = request.commission_pct,
        )

        risk = RiskManager(portfolio=portfolio)

        regime = RegimeDetector(
            data_handler  = data,
            symbol        = request.symbols[0],
            lookback      = 252,
            retrain_every = 126,
        )

        # ── Build strategy list ───────────────────────────────────────────────
        strategy_list = [regime]

        if request.strategy in ("ml", "ml+pairs"):
            ml = MLSignalStrategy(
                data_handler    = data,
                symbols         = request.symbols,
                lookback        = 252,
                retrain_every   = 63,
                threshold       = request.ml_threshold,
                regime_detector = regime,
            )
            strategy_list.append(ml)

        if request.strategy in ("momentum", "ml+momentum"):
            momentum = MomentumStrategy(
                data_handler    = data,
                symbols         = request.symbols,
                lookback        = request.momentum_lookback,
                enter_threshold = request.momentum_enter_threshold,
                exit_threshold  = request.momentum_exit_threshold,
                min_hold_bars   = request.momentum_min_hold_bars,
                rebalance_every = request.momentum_rebalance_every,
                regime_detector = regime,
            )
            strategy_list.append(momentum)

        if request.strategy in ("pairs", "ml+pairs"):
            pairs = KalmanPairsStrategy(
                data_handler  = data,
                pair          = ("PEP", "CVX"),
                z_score_entry = 2.0,
                z_score_exit  = 0.5,
            )
            strategy_list.append(pairs)
        
        # ── Run engine ────────────────────────────────────────────────────────
        engine = Engine(
            data_handler      = data,
            strategies        = strategy_list,
            portfolio         = portfolio,
            execution_handler = execution,
            risk_manager      = risk,
        )

        engine.run()

        # ── Deduplicate equity curve (one point per date) ─────────────────────
        seen = {}
        for e in portfolio.equity_curve:
            date_key = e["timestamp"].strftime("%Y-%m-%d") if hasattr(e["timestamp"], "strftime") else str(e["timestamp"])[:10]
            seen[date_key] = e
        deduped_equity = list(seen.values())

        # ── Format equity curve ───────────────────────────────────────────────
        from pathlib import Path
        from data.cache import load_from_cache

# Equal-weighted buy and hold across all symbols
        bh_series = {}
        for sym in request.symbols:
            df = load_from_cache(sym, request.start_date, request.end_date, Path(".cache/data"))
            if df is not None and "close" in df.columns:
                close = df["close"]
                bh_series[sym] = close / float(close.iloc[0])  # normalise each to 1.0

        if bh_series:
            import pandas as pd
            bh_combined = pd.DataFrame(bh_series).mean(axis=1)  # equal weight average
            bh_map = {str(d.date()): round(float(v) * request.initial_capital, 2)
                      for d, v in bh_combined.items()}
        else:
            bh_map = {}

        equity_curve = [
            EquityPoint(
                timestamp    = e["timestamp"].isoformat() if hasattr(e["timestamp"], "isoformat") else str(e["timestamp"]),
                total_equity = round(e["total_equity"], 2),
                cash         = round(e["cash"], 2),
                market_value = round(e["market_value"], 2),
                buyhold_norm = bh_map.get(
                    e["timestamp"].strftime("%Y-%m-%d") if hasattr(e["timestamp"], "strftime") else str(e["timestamp"])[:10],
                    request.initial_capital
                ),
            )
            for e in deduped_equity
        ]

        # ── Format regime history ─────────────────────────────────────────────
        regime_history = [
            RegimePoint(
                timestamp     = r["timestamp"].isoformat() if hasattr(r["timestamp"], "isoformat") else str(r["timestamp"]),
                regime        = r["regime"],
                prob_bull     = r["prob_bull"],
                prob_sideways = r["prob_sideways"],
                prob_bear     = r["prob_bear"],
            )
            for r in regime.regime_history
        ]

        duration = (datetime.utcnow() - start_time).total_seconds()

        return BacktestResponse(
            status         = "success",
            duration_secs  = round(duration, 2),
            metrics        = portfolio.metrics,
            equity_curve   = equity_curve,
            regime_history = regime_history,
        )

    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
