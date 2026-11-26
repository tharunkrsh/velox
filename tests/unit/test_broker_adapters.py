"""End-to-end tests for the MT5 (UTF-16/tab and comma) and Tradovate adapters."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from velox.app.cli import app
from velox.domain.calendar import SessionCalendar
from velox.domain.daybook import build_daybook
from velox.domain.trades import TradeLogError
from velox.ingest import load_trades

FIX = Path(__file__).parent.parent / "fixtures"
PRAGUE = SessionCalendar(timezone="Europe/Prague", rollover=dt.time(0, 0))
CHICAGO_17 = SessionCalendar(timezone="America/Chicago", rollover=dt.time(17, 0))


@pytest.mark.parametrize("name", ["mt5_positions_utf16.tsv", "mt5_positions_comma.csv"])
def test_mt5_desktop_exports(name):
    log = load_trades(FIX / name, fmt="mt5", timezone="Europe/Prague")
    assert len(log) == 4  # totals row dropped
    np.testing.assert_allclose(log.pnl, [198 - 7, -270 - 3.5, 49, 1251 - 21 - 9.3])
    np.testing.assert_allclose(log.units, [1, 0.5, 2, 3])
    assert list(log.symbol) == ["EURUSD", "XAUUSD", "US100.cash", "GBPUSD"]
    assert not log.has_risk and not log.has_mae  # blank S/L, no MAE in MT5
    # 23:40 -> 00:20 Prague hold counts on the day it closed
    book = build_daybook(log, PRAGUE)
    assert [str(d) for d in book.days] == ["2026-03-02", "2026-03-03", "2026-03-04"]
    np.testing.assert_allclose(book.daily_pnl(), [191 - 273.5, 49, 1220.7])


def test_mt5_utf16_and_comma_variants_are_identical():
    a = load_trades(FIX / "mt5_positions_utf16.tsv", fmt="mt5", timezone="Europe/Prague")
    b = load_trades(FIX / "mt5_positions_comma.csv", fmt="mt5", timezone="Europe/Prague")
    assert a.to_frame().equals(b.to_frame())


def test_tradovate_performance_report():
    log = load_trades(FIX / "tradovate_performance.csv", fmt="tradovate",
                      timezone="America/Chicago", commission_per_unit=1.24)
    assert len(log) == 4  # two partial fills merged into one trade
    np.testing.assert_allclose(log.units, [2, 2, 3, 2])
    np.testing.assert_allclose(log.pnl, [42.5 - 2.48, -62.5 - 2.48, 118.5 - 3.72,
                                         -1012.5 - 2.48])
    # the short trade sold first: opened 09:58:12 CST = 15:58:12 UTC
    assert log.open_time[1].strftime("%H:%M:%S") == "15:58:12"
    assert log.close_time[1].strftime("%H:%M:%S") == "16:05:00"
    # 16:50 -> 17:20 CT hold belongs to the next Topstep trading day
    book = build_daybook(log, CHICAGO_17)
    assert [str(d) for d in book.days] == ["2026-03-02", "2026-03-03"]
    assert np.diff(book.offsets).tolist() == [2, 2]


def test_tradovate_gross_pnl_without_commission():
    log = load_trades(FIX / "tradovate_performance.csv", fmt="tradovate",
                      timezone="America/Chicago")
    assert log.pnl.sum() == pytest.approx(42.5 - 62.5 + 118.5 - 1012.5)


def test_broker_formats_require_timezone_and_columns():
    with pytest.raises(ValueError, match="timezone"):
        load_trades(FIX / "tradovate_performance.csv", fmt="tradovate")
    with pytest.raises(TradeLogError, match="Tradovate"):
        load_trades(FIX / "mt5_positions_comma.csv", fmt="tradovate", timezone="UTC")


def test_cli_replay_with_broker_formats():
    runner = CliRunner()
    res = runner.invoke(app, ["replay", str(FIX / "tradovate_performance.csv"), "-f", "tradovate",
                              "--tz", "America/Chicago", "-p", "topstep-50k",
                              "--commission-per-unit", "1.24"])
    assert res.exit_code == 0, res.output
    assert "NOT FINISHED" in res.output
    res = runner.invoke(app, ["inspect", str(FIX / "mt5_positions_utf16.tsv"), "-f", "mt5",
                              "--tz", "Europe/Prague", "-p", "ftmo-2step"])
    assert res.exit_code == 0, res.output
    assert "4 trades over 3 trading days" in res.output
