# VELOX
### Algorithmic Trading Research Framework

> Backtesting engine with HMM regime detection, walk-forward ML signals, and Kalman filter pairs trading, served via FastAPI with a React dashboard.
![Python](https://img.shields.io/badge/Python-3.13-00ff88?style=flat-square&logo=python&logoColor=white&labelColor=0a0a0f)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00ff88?style=flat-square&logo=fastapi&logoColor=white&labelColor=0a0a0f)
![React](https://img.shields.io/badge/React-18-00ff88?style=flat-square&logo=react&logoColor=white&labelColor=0a0a0f)
![LightGBM](https://img.shields.io/badge/LightGBM-ML-00ff88?style=flat-square&labelColor=0a0a0f)
![Docker](https://img.shields.io/badge/Docker-ready-00ff88?style=flat-square&logo=docker&logoColor=white&labelColor=0a0a0f)

## What is VELOX?

VELOX is a full-stack algorithmic trading research platform built from scratch. It runs bar-by-bar backtests on historical price data, detects market regimes using a Hidden Markov Model, generates trading signals via a walk-forward LightGBM classifier, and displays results through a REST API and React dashboard.

## Architecture

<img width="2816" height="1536" alt="Gemini_Generated_Image_qxjia8qxjia8qxji" src="https://github.com/user-attachments/assets/37618ed0-c138-499f-ba35-2b60f0246319" />

## How it looks

React dashboard: choose your strategy, timeframe, and tickers, then adjust commission, slippage, and starting capital.

<img width="956" height="464" alt="dashboard" src="https://github.com/user-attachments/assets/16a451a7-a45e-41be-b0bb-a9d03649a19a" />

Velox then shows your strategy performance against a buy-and-hold benchmark.

<img width="947" height="462" alt="results1" src="https://github.com/user-attachments/assets/11636e16-0738-427c-9cdf-b91558cea70c" />

The results dashboard shows total return, Sharpe ratio, max drawdown, total trades, annual return, annual volatility, Calmar ratio, final equity, a drawdown chart, and a market regime probability chart.

<img width="947" height="452" alt="results2" src="https://github.com/user-attachments/assets/fb3c4c3a-ae3a-440a-a439-ba2c5fefe2ff" />

### How it works: Engine Design

VELOX uses a two-level event loop identical to production trading systems:

1. **Outer loop** iterates through historical bars, emitting a `MarketEvent` on each bar  
2. **Inner loop** drains the event queue; strategies consume `MarketEvent`, emit `SignalEvent`, the risk manager validates, produces an `OrderEvent`, the execution handler fills it, and the portfolio updates on `FillEvent`

This architecture prevents lookahead bias at the structural level. No strategy can ever see future prices.

## Strategies

### 1. HMM Regime Detector (`signals/regime.py`)
A Gaussian Hidden Markov Model with 3 latent states (bull / sideways / bear), trained on rolling 252-bar windows and retrained quarterly. States are labelled by mean return ranking.

All downstream strategies gate on the current regime. Momentum and ML signals are suppressed in bear regimes, reducing drawdown without sacrificing bull-market returns.

### 2. Walk-Forward LightGBM Signal (`signals/ml_signal.py`)
A gradient-boosted classifier trained on rolling windows with strict temporal separation between train and test sets (no lookahead bias). Features include:

- Multi-timeframe returns (1, 5, 10, 20 bars)
- Rolling volatility (10, 20 bars)
- Distance from moving averages (10, 20, 50 bars)
- Relative Strength Index (14 bars)
- Volume ratio

The model is retrained every 21 bars (monthly) on the most recent 252 bars of data. Signals are only fired when predicted probability exceeds a configurable threshold (default 0.6).

### 3. Time-Series Momentum (`signals/momentum.py`)
Jegadeesh & Titman (1993) cross-sectional momentum with a 40-bar lookback and 2% threshold. Regime-gated ; no trades in bear markets.

### 4. Kalman Filter Pairs Trading (`signals/pairs.py`)
Market-neutral statistical arbitrage on the PEP/CVX pair (Engle-Granger cointegration p=0.0114). The Kalman filter dynamically estimates the hedge ratio as a latent variable that evolves as a random walk.

**State equation:** `β_t = β_{t-1} + w_t` (hedge ratio drifts as random walk)  
**Observation:** `y_t = β_t · x_t + v_t` (prices observed with noise)

Entry at ±2σ spread divergence, exit at ±0.5σ reversion.

## Performance (ML Strategy, 2020–2023)

| Metric | VELOX | Buy & Hold (equal-weighted) |
|--------|-------|--------------------------|
| Total Return | 25.4% | ~45% |
| Ann. Return | 3.63% | ~10% |
| Ann. Volatility | 4.3% | ~22% |
| Sharpe Ratio | 0.844 | ~0.45 |
| Max Drawdown | -4.92% | -30%+ (2022) |
| Calmar Ratio | 0.739 | ~0.33 |
| Total Trades | 594 | 1 |

VELOX targets **risk-adjusted returns** rather than raw performance. During the 2022 bear market (S&P -19%), VELOX drawdown was capped at -4.92% due to HMM regime gating suppressing all signals in bear conditions.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| ML | LightGBM, hmmlearn, scikit-learn |
| Data | yfinance, pandas, pyarrow (Parquet cache) |
| Statistics | statsmodels (cointegration tests) |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | React 18, Vite, Recharts, Axios |
| Package mgmt | uv |
| Deployment | Docker, docker-compose |

## Project Structure

```
velox/
├── core/
│   ├── engine.py          # Two-level event loop
│   ├── events.py          # Event dataclasses (Market/Signal/Order/Fill)
│   ├── portfolio.py       # Position sizing, P&L, equity curve
│   ├── execution.py       # Simulated execution with slippage + commission
│   └── risk.py            # Position limits, drawdown circuit breaker
├── data/
│   ├── historical.py      # yfinance data handler with lookahead prevention
│   ├── cache.py           # Parquet caching layer
│   └── base.py            # Abstract base class
├── signals/
│   ├── regime.py          # HMM regime detector
│   ├── ml_signal.py       # Walk-forward LightGBM classifier
│   ├── momentum.py        # Time-series momentum
│   └── pairs.py           # Kalman filter pairs trading
├── api/
│   └── main.py            # FastAPI backend
├── dashboard/
│   └── src/App.jsx        # React dashboard
├── research/
│   ├── tearsheet.py       # Performance report generator
│   └── visualizer.py      # Plotly HTML visualizer
├── Dockerfile             # API container
├── dashboard/Dockerfile   # Dashboard container
├── docker-compose.yml     # Orchestration
└── run_backtest.py        # Standalone backtest entry point
```

## Quickstart

### Prerequisites
- Python 3.13+
- Node.js 22+
- [uv](https://github.com/astral-sh/uv)

### 1. Clone and install

```bash
git clone https://github.com/tharunkrsh/velox.git
cd velox
uv sync
```

### 2. Run a backtest (command line)

```bash
python run_backtest.py
```

### 3. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Start the dashboard

```bash
cd dashboard
npm install
npm run dev
```

Open `http://localhost:5173`, configure parameters, and hit **RUN BACKTEST**.

## Docker

VELOX is fully containerised. With Docker installed, the entire stack spins up with a single command — no Python, Node, or dependency setup required.

```bash
docker compose up --build
```

This starts two containers: the FastAPI backend on port 8000 and the React dashboard on port 5173. The price cache is mounted as a volume so downloaded data persists between runs.

To run in the background:

```bash
docker compose up --build -d
```

To stop:

```bash
docker compose down
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/strategies` | GET | List available strategies |
| `/backtest` | POST | Run a full backtest |

Auto-generated docs at `http://localhost:8000/docs`.

### Example request

```bash
curl -X POST http://localhost:8000/backtest \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT", "GOOGL", "PEP", "CVX"],
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 100000,
    "strategy": "ml",
    "ml_threshold": 0.6
  }'
```

## Design Decisions

**Why a bar-by-bar event loop?** Production trading systems process events asynchronously ; orders arrive out of order, fills are uncertain, and risk checks must happen in real time. Building VELOX with the same architecture means the backtesting logic is directly comparable to live trading.

**Why walk-forward ML?** I designed the walk-forward ML algorithm to avoid a common quant research mistake: evaluating strategies on the same data used to develop them, which can inflate performance through overfitting.. Walk-forward validation enforces strict temporal separation ; the model never sees future data during training, making the backtest results honest.

**Why Kalman filter for pairs?** Static OLS hedge ratios assume the relationship between two assets is fixed. In reality it drifts ; regime changes, current events and macro shifts all affect the cointegration relationship. The Kalman filter treats the hedge ratio as a latent state that evolves over time, updating optimally on each new observation.

**Why HMM for regimes?** Markets exhibit distinct statistical properties in different regimes ; volatility, autocorrelation, and return distributions all change. The HMM learns these latent states from return data alone, without requiring labelled training data. The Viterbi algorithm then assigns the most likely regime sequence given the observed returns. This allows for maximum returns by using each strategy during its optimal regime.

## Known Limitations

- Pairs strategy requires periodic re-screening for cointegration breakdown
- ML signal does not incorporate fundamental data or alternative data sources
- Backtests assume perfect liquidity ; real execution would face market impact
- Timestamp deduplication edge case in equity curve (tracked, fix pending)

*Built with Python, FastAPI, and React.*
