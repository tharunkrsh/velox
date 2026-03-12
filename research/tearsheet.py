"""
tearsheet.py - Performance report generator for VELOX.

Generates a full quantitative tearsheet from a completed
backtest. Saves results as both a text report and CSV
for the dashboard to consume later.

Metrics included:
    Returns:    Total, annualised, monthly breakdown
    Risk:       Volatility, VaR, CVaR, max drawdown
    Ratios:     Sharpe, Sortino, Calmar
    Trading:    Win rate, avg win/loss, profit factor
"""

import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Tearsheet:

    def __init__(self, portfolio, output_dir: str = "research/output"):
        self.portfolio  = portfolio
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, strategy_name: str = "velox") -> dict:
        """
        Generate full tearsheet from portfolio data.
        Saves report to file and returns metrics dict.
        """
        if not self.portfolio.equity_curve:
            logger.warning("No equity curve data.")
            return {}

        df     = pd.DataFrame(self.portfolio.equity_curve)
        equity = df["total_equity"].astype(float)
        returns = equity.pct_change().dropna()

        metrics = {}

        # ── Return metrics ────────────────────────────────────────────────────
        metrics["total_return_pct"]   = round((equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2)
        metrics["ann_return_pct"]     = round(self._annualised_return(equity) * 100, 2)
        metrics["best_day_pct"]       = round(returns.max() * 100, 2)
        metrics["worst_day_pct"]      = round(returns.min() * 100, 2)
        metrics["positive_days_pct"]  = round((returns > 0).mean() * 100, 2)

        # ── Risk metrics ──────────────────────────────────────────────────────
        ann_vol = returns.std() * np.sqrt(252)
        metrics["ann_volatility_pct"] = round(ann_vol * 100, 2)
        metrics["max_drawdown_pct"]   = round(self._max_drawdown(equity) * 100, 2)
        metrics["avg_drawdown_pct"]   = round(self._avg_drawdown(equity) * 100, 2)
        metrics["var_95_pct"]         = round(np.percentile(returns, 5) * 100, 2)
        metrics["cvar_95_pct"]        = round(returns[returns <= np.percentile(returns, 5)].mean() * 100, 2)

        # ── Risk-adjusted ratios ──────────────────────────────────────────────
        ann_ret = self._annualised_return(equity)
        metrics["sharpe_ratio"]  = round(self._sharpe(returns), 3)
        metrics["sortino_ratio"] = round(self._sortino(returns), 3)
        metrics["calmar_ratio"]  = round(self._calmar(equity), 3)

        # ── Trading metrics ───────────────────────────────────────────────────
        fills = self.portfolio.fill_history
        metrics["total_trades"]  = len(fills)
        metrics["total_commission"] = round(sum(f.commission for f in fills), 2)

        trade_pnls = self._calculate_trade_pnls()
        if trade_pnls:
            wins  = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p < 0]
            metrics["win_rate_pct"]    = round(len(wins) / len(trade_pnls) * 100, 2)
            metrics["avg_win"]         = round(np.mean(wins), 2) if wins else 0
            metrics["avg_loss"]        = round(np.mean(losses), 2) if losses else 0
            metrics["profit_factor"]   = round(
                sum(wins) / abs(sum(losses)), 3
            ) if losses else float("inf")

        # ── Save outputs ──────────────────────────────────────────────────────
        self._save_report(metrics, strategy_name)
        self._save_equity_curve(df, strategy_name)

        return metrics

    # ─── Metric Calculations ──────────────────────────────────────────────────

    def _annualised_return(self, equity: pd.Series) -> float:
        n_days       = len(equity)
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        return (1 + total_return) ** (252 / n_days) - 1

    def _max_drawdown(self, equity: pd.Series) -> float:
        rolling_max = equity.cummax()
        drawdown    = (equity - rolling_max) / rolling_max
        return drawdown.min()

    def _avg_drawdown(self, equity: pd.Series) -> float:
        rolling_max = equity.cummax()
        drawdown    = (equity - rolling_max) / rolling_max
        return drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0.0

    def _sharpe(self, returns: pd.Series, risk_free: float = 0.02) -> float:
        excess = returns - risk_free / 252
        if excess.std() == 0:
            return 0.0
        return excess.mean() / excess.std() * np.sqrt(252)

    def _sortino(self, returns: pd.Series, risk_free: float = 0.02) -> float:
        """
        Sortino ratio — like Sharpe but only penalises downside volatility.
        Better measure for strategies that have asymmetric return profiles.
        """
        excess         = returns - risk_free / 252
        downside_std   = returns[returns < 0].std()
        if downside_std == 0:
            return 0.0
        return excess.mean() / downside_std * np.sqrt(252)

    def _calmar(self, equity: pd.Series) -> float:
        ann_ret    = self._annualised_return(equity)
        max_dd     = self._max_drawdown(equity)
        if max_dd == 0:
            return 0.0
        return ann_ret / abs(max_dd)

    def _calculate_trade_pnls(self) -> list:
        """
        Calculate P&L for each completed round-trip trade.
        Pairs buy fills with subsequent sell fills per symbol.
        """
        from core.events import OrderDirection

        pnls        = []
        open_trades = {}

        for fill in self.portfolio.fill_history:
            symbol = fill.symbol

            if fill.direction == OrderDirection.BUY:
                open_trades[symbol] = fill
            else:
                if symbol in open_trades:
                    entry = open_trades.pop(symbol)
                    pnl   = (fill.fill_price - entry.fill_price) * fill.quantity
                    pnls.append(pnl)

        return pnls

    # ─── Output ───────────────────────────────────────────────────────────────

    def _save_report(self, metrics: dict, name: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path      = self.output_dir / f"{name}_tearsheet_{timestamp}.txt"

        lines = [
            "═" * 50,
            f"  VELOX TEARSHEET — {name.upper()}",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "═" * 50,
            "",
            "  RETURNS",
            f"  {'Total Return':<25} {metrics.get('total_return_pct', 'N/A'):>8}%",
            f"  {'Annualised Return':<25} {metrics.get('ann_return_pct', 'N/A'):>8}%",
            f"  {'Best Day':<25} {metrics.get('best_day_pct', 'N/A'):>8}%",
            f"  {'Worst Day':<25} {metrics.get('worst_day_pct', 'N/A'):>8}%",
            f"  {'Positive Days':<25} {metrics.get('positive_days_pct', 'N/A'):>8}%",
            "",
            "  RISK",
            f"  {'Annualised Volatility':<25} {metrics.get('ann_volatility_pct', 'N/A'):>8}%",
            f"  {'Max Drawdown':<25} {metrics.get('max_drawdown_pct', 'N/A'):>8}%",
            f"  {'Avg Drawdown':<25} {metrics.get('avg_drawdown_pct', 'N/A'):>8}%",
            f"  {'VaR (95%)':<25} {metrics.get('var_95_pct', 'N/A'):>8}%",
            f"  {'CVaR (95%)':<25} {metrics.get('cvar_95_pct', 'N/A'):>8}%",
            "",
            "  RATIOS",
            f"  {'Sharpe Ratio':<25} {metrics.get('sharpe_ratio', 'N/A'):>8}",
            f"  {'Sortino Ratio':<25} {metrics.get('sortino_ratio', 'N/A'):>8}",
            f"  {'Calmar Ratio':<25} {metrics.get('calmar_ratio', 'N/A'):>8}",
            "",
            "  TRADING",
            f"  {'Total Trades':<25} {metrics.get('total_trades', 'N/A'):>8}",
            f"  {'Total Commission':<25} ${metrics.get('total_commission', 'N/A'):>7}",
            f"  {'Win Rate':<25} {metrics.get('win_rate_pct', 'N/A'):>8}%",
            f"  {'Avg Win':<25} ${metrics.get('avg_win', 'N/A'):>7}",
            f"  {'Avg Loss':<25} ${metrics.get('avg_loss', 'N/A'):>7}",
            f"  {'Profit Factor':<25} {metrics.get('profit_factor', 'N/A'):>8}",
            "",
            "═" * 50,
        ]

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Tearsheet saved: {path}")

    def _save_equity_curve(self, df: pd.DataFrame, name: str) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path      = self.output_dir / f"{name}_equity_{timestamp}.csv"
        df.to_csv(path, index=False)
        logger.info(f"Equity curve saved: {path}")