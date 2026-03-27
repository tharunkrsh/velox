"""
Evaluate one fixed momentum config across multiple 4-year periods.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.tune_momentum import Config, run_once


def main() -> None:
    symbols = ["AAPL", "MSFT", "GOOGL", "PEP", "CVX"]
    capital = 100_000.0
    slippage_pct = 0.001
    commission_pct = 0.001

    cfg = Config(
        lookback=40,
        enter=0.04,
        exit=-0.02,
        min_hold=10,
        rebalance=5,
    )

    periods = [
        ("2010-01-01", "2013-12-31"),
        ("2012-01-01", "2015-12-31"),
        ("2014-01-01", "2017-12-31"),
        ("2016-01-01", "2019-12-31"),
        ("2018-01-01", "2021-12-31"),
        ("2020-01-01", "2023-12-31"),
    ]

    rows = []
    for start_date, end_date in periods:
        r = run_once(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            capital=capital,
            slippage_pct=slippage_pct,
            commission_pct=commission_pct,
            cfg=cfg,
        )
        outperf = round(r["total_return_pct"] - r["buyhold_return_pct"], 2)
        rows.append((start_date, end_date, r["total_return_pct"], r["buyhold_return_pct"], outperf, r["sharpe"], r["max_drawdown_pct"], r["trades"], r["final_equity"]))

    print("Fixed momentum config:")
    print("lookback=40, enter=0.04, exit=-0.02, min_hold=10, rebalance=5")
    print("\nResults by 4-year window")
    print("start       end         mom_ret%  bnh_ret%  outperf%  sharpe  max_dd%  trades  final_equity")
    for row in rows:
        print(
            f"{row[0]}  {row[1]}  "
            f"{row[2]:7.2f}  {row[3]:8.2f}  {row[4]:8.2f}  "
            f"{row[5]:6.3f}  {row[6]:7.2f}  {row[7]:6d}  {row[8]:11.2f}"
        )


if __name__ == "__main__":
    main()

