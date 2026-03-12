"""
visualizer.py - Interactive HTML chart generator for VELOX.

Generates a standalone HTML file with:
    - Equity curve vs buy-and-hold benchmark
    - Price chart with regime coloring
    - Regime probability over time
    - Drawdown chart

Uses Plotly via CDN — no extra installs needed.
Open the output file in any browser.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Visualizer:

    def __init__(self, portfolio, regime_detector=None, data_handler=None):
        self.portfolio = portfolio
        self.regime    = regime_detector
        self.data      = data_handler

    def generate(
        self,
        symbol:     str  = "AAPL",
        output_dir: str  = "research/output",
        name:       str  = "velox",
    ) -> str:
        """
        Generate interactive HTML chart.
        Returns path to the output file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path      = output_dir / f"{name}_chart_{timestamp}.html"

        # ── Build data ────────────────────────────────────────────────────────
        equity_df  = self._build_equity_df()
        regime_df  = self._build_regime_df()
        price_df   = self._build_price_df(symbol)

        # ── Build HTML ────────────────────────────────────────────────────────
        html = self._build_html(equity_df, regime_df, price_df, symbol)
        path.write_text(html, encoding="utf-8")

        logger.info(f"Chart saved: {path}")
        print(f"\n📊 Chart saved: {path}")
        print(f"   Open in browser: file:///{path.absolute()}")

        return str(path)

    # ─── Data Builders ────────────────────────────────────────────────────────

    def _build_equity_df(self) -> pd.DataFrame:
        if not self.portfolio.equity_curve:
            return pd.DataFrame()

        df = pd.DataFrame(self.portfolio.equity_curve)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Normalise to 100 for easy comparison
        df["equity_norm"] = df["total_equity"] / df["total_equity"].iloc[0] * 100

        # Drawdown
        rolling_max     = df["total_equity"].cummax()
        df["drawdown"]  = (df["total_equity"] - rolling_max) / rolling_max * 100

        return df

    def _build_regime_df(self) -> pd.DataFrame:
        if self.regime is None:
            return pd.DataFrame()
        return self.regime.get_regime_df()

    def _build_price_df(self, symbol: str) -> pd.DataFrame:
        if self.data is None:
            return pd.DataFrame()

        bars = self.data.latest_bars.get(symbol, [])
        if not bars:
            return pd.DataFrame()

        dates  = [b[0] for b in bars]
        closes = [float(b[1]["close"]) for b in bars]

        df = pd.DataFrame({"date": dates, "close": closes})
        df["date"] = pd.to_datetime(df["date"])

        # Normalise to 100
        df["close_norm"] = df["close"] / df["close"].iloc[0] * 100

        return df

    # ─── HTML Builder ─────────────────────────────────────────────────────────

    def _build_html(
        self,
        equity_df:  pd.DataFrame,
        regime_df:  pd.DataFrame,
        price_df:   pd.DataFrame,
        symbol:     str,
    ) -> str:

        # Convert DataFrames to JSON for Plotly
        equity_dates  = equity_df["timestamp"].dt.strftime("%Y-%m-%d").tolist() if not equity_df.empty else []
        equity_values = equity_df["equity_norm"].round(2).tolist() if not equity_df.empty else []
        drawdown_values = equity_df["drawdown"].round(2).tolist() if not equity_df.empty else []

        price_dates  = price_df["date"].dt.strftime("%Y-%m-%d").tolist() if not price_df.empty else []
        price_values = price_df["close_norm"].round(2).tolist() if not price_df.empty else []

        # Regime shapes for background coloring
        regime_shapes = []
        regime_colors = {"bull": "rgba(0,200,100,0.15)", "bear": "rgba(220,50,50,0.15)", "sideways": "rgba(255,200,0,0.12)"}

        if not regime_df.empty:
            regime_df["timestamp"] = pd.to_datetime(regime_df["timestamp"])
            prev_regime = None
            start_date  = None

            for _, row in regime_df.iterrows():
                if row["regime"] != prev_regime:
                    if prev_regime is not None:
                        regime_shapes.append({
                            "type": "rect",
                            "xref": "x", "yref": "paper",
                            "x0": start_date, "x1": row["timestamp"].strftime("%Y-%m-%d"),
                            "y0": 0, "y1": 1,
                            "fillcolor": regime_colors.get(prev_regime, "rgba(200,200,200,0.1)"),
                            "line": {"width": 0},
                            "layer": "below",
                        })
                    prev_regime = row["regime"]
                    start_date  = row["timestamp"].strftime("%Y-%m-%d")

        regime_shapes_json = json.dumps(regime_shapes)

        # Regime probability data
        prob_dates     = []
        prob_bull      = []
        prob_bear      = []
        prob_sideways  = []

        if not regime_df.empty:
            prob_dates    = regime_df["timestamp"].dt.strftime("%Y-%m-%d").tolist()
            prob_bull     = regime_df["prob_bull"].tolist()
            prob_bear     = regime_df["prob_bear"].tolist()
            prob_sideways = regime_df["prob_sideways"].tolist()

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VELOX — Strategy Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #0a0a0f;
            color: #e0e0e0;
            font-family: 'SF Mono', 'Fira Code', monospace;
            padding: 24px;
        }}
        h1 {{
            font-size: 28px;
            font-weight: 700;
            letter-spacing: 4px;
            color: #00ff88;
            margin-bottom: 4px;
        }}
        .subtitle {{
            color: #666;
            font-size: 13px;
            margin-bottom: 32px;
            letter-spacing: 2px;
        }}
        .legend {{
            display: flex;
            gap: 24px;
            margin-bottom: 24px;
            font-size: 12px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}
        .bull   {{ background: rgba(0,200,100,0.6); }}
        .bear   {{ background: rgba(220,50,50,0.6); }}
        .sideways {{ background: rgba(255,200,0,0.6); }}
        .chart {{
            background: #0f0f1a;
            border: 1px solid #1e1e2e;
            border-radius: 8px;
            margin-bottom: 16px;
            padding: 16px;
        }}
        .chart-title {{
            font-size: 11px;
            letter-spacing: 2px;
            color: #888;
            margin-bottom: 12px;
            text-transform: uppercase;
        }}
    </style>
</head>
<body>

    <h1>VELOX</h1>
    <div class="subtitle">ALGORITHMIC TRADING RESEARCH FRAMEWORK — STRATEGY DASHBOARD</div>

    <div class="legend">
        <div class="legend-item"><div class="dot bull"></div><span>Bull Regime</span></div>
        <div class="legend-item"><div class="dot sideways"></div><span>Sideways Regime</span></div>
        <div class="legend-item"><div class="dot bear"></div><span>Bear Regime</span></div>
    </div>

    <div class="chart">
        <div class="chart-title">Equity Curve vs {symbol} Buy & Hold (Normalised to 100)</div>
        <div id="equity-chart"></div>
    </div>

    <div class="chart">
        <div class="chart-title">Drawdown</div>
        <div id="drawdown-chart"></div>
    </div>

    <div class="chart">
        <div class="chart-title">Regime Probabilities</div>
        <div id="regime-chart"></div>
    </div>

<script>
const equityDates  = {json.dumps(equity_dates)};
const equityValues = {json.dumps(equity_values)};
const drawdownValues = {json.dumps(drawdown_values)};
const priceDates   = {json.dumps(price_dates)};
const priceValues  = {json.dumps(price_values)};
const probDates    = {json.dumps(prob_dates)};
const probBull     = {json.dumps(prob_bull)};
const probBear     = {json.dumps(prob_bear)};
const probSideways = {json.dumps(prob_sideways)};
const regimeShapes = {regime_shapes_json};

const darkLayout = {{
    paper_bgcolor: '#0f0f1a',
    plot_bgcolor:  '#0f0f1a',
    font:          {{ color: '#888', family: 'SF Mono, Fira Code, monospace', size: 11 }},
    xaxis:         {{ gridcolor: '#1e1e2e', linecolor: '#1e1e2e', zeroline: false }},
    yaxis:         {{ gridcolor: '#1e1e2e', linecolor: '#1e1e2e', zeroline: false }},
    margin:        {{ l: 60, r: 20, t: 20, b: 40 }},
    showlegend:    true,
    legend:        {{ bgcolor: 'rgba(0,0,0,0)', font: {{ size: 11 }} }},
    shapes:        regimeShapes,
}};

// ── Equity chart ──────────────────────────────────────────────────────────────
Plotly.newPlot('equity-chart', [
    {{
        x: equityDates, y: equityValues,
        type: 'scatter', mode: 'lines',
        name: 'VELOX Strategy',
        line: {{ color: '#00ff88', width: 2 }},
    }},
    {{
        x: priceDates, y: priceValues,
        type: 'scatter', mode: 'lines',
        name: '{symbol} Buy & Hold',
        line: {{ color: '#4488ff', width: 1.5, dash: 'dot' }},
    }},
], {{...darkLayout, height: 300}}, {{responsive: true}});

// ── Drawdown chart ────────────────────────────────────────────────────────────
Plotly.newPlot('drawdown-chart', [
    {{
        x: equityDates, y: drawdownValues,
        type: 'scatter', mode: 'lines', fill: 'tozeroy',
        name: 'Drawdown',
        line:       {{ color: '#ff4466', width: 1.5 }},
        fillcolor:  'rgba(255,68,102,0.15)',
    }},
], {{...darkLayout, height: 200, shapes: regimeShapes}}, {{responsive: true}});

// ── Regime probability chart ──────────────────────────────────────────────────
Plotly.newPlot('regime-chart', [
    {{
        x: probDates, y: probBull,
        type: 'scatter', mode: 'lines', fill: 'tozeroy',
        name: 'Bull',
        line: {{ color: '#00cc66', width: 1 }},
        fillcolor: 'rgba(0,204,102,0.2)',
        stackgroup: 'one',
    }},
    {{
        x: probDates, y: probSideways,
        type: 'scatter', mode: 'lines', fill: 'tozeroy',
        name: 'Sideways',
        line: {{ color: '#ffcc00', width: 1 }},
        fillcolor: 'rgba(255,204,0,0.2)',
        stackgroup: 'one',
    }},
    {{
        x: probDates, y: probBear,
        type: 'scatter', mode: 'lines', fill: 'tozeroy',
        name: 'Bear',
        line: {{ color: '#ff4444', width: 1 }},
        fillcolor: 'rgba(255,68,68,0.2)',
        stackgroup: 'one',
    }},
], {{...darkLayout, height: 200, shapes: []}}, {{responsive: true}});

</script>
</body>
</html>"""

        return html