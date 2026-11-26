from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from velox.domain import stats
from velox.domain.calendar import SessionCalendar
from velox.domain.daybook import build_daybook
from velox.domain.sizing import IntratradeModel, SizingModel, adverse_amount
from velox.domain.trades import TradeLog, TradeLogError
from velox.presets import list_presets, load_preset

# ─── calendar ────────────────────────────────────────────────────────────────


def _utc(*stamps: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(stamps).tz_localize("UTC")


def test_topstep_evening_rollover_assigns_sunday_night_to_monday():
    cal = SessionCalendar(timezone="America/Chicago", rollover=dt.time(17, 0))
    days = cal.trading_day(_utc(
        "2024-01-14 23:30",  # Sun 17:30 CST -> Mon
        "2024-01-15 21:30",  # Mon 15:30 CST -> Mon
        "2024-01-15 23:05",  # Mon 17:05 CST -> Tue
        "2024-07-15 21:59",  # Mon 16:59 CDT -> Mon (DST)
        "2024-07-15 22:01",  # Mon 17:01 CDT -> Tue
    ))
    assert [str(d) for d in days] == [
        "2024-01-15", "2024-01-15", "2024-01-16", "2024-07-15", "2024-07-16"]


def test_ftmo_midnight_prague_rollover_with_dst():
    cal = SessionCalendar(timezone="Europe/Prague", rollover=dt.time(0, 0))
    days = cal.trading_day(_utc(
        "2024-01-15 22:59",  # 23:59 CET
        "2024-01-15 23:01",  # 00:01 CET next day
        "2024-07-15 22:30",  # 00:30 CEST next day
    ))
    assert [str(d) for d in days] == ["2024-01-15", "2024-01-16", "2024-07-16"]


def test_unknown_timezone_rejected():
    with pytest.raises(ValueError):
        SessionCalendar(timezone="Mars/Olympus")


# ─── trade log, sizing, daybook ──────────────────────────────────────────────


def _log(pnl, units=None, risk=None, mae=None, closes=None) -> TradeLog:
    n = len(pnl)
    closes = closes or [f"2024-01-{d:02d} 15:00" for d in range(1, n + 1)]
    close = pd.DatetimeIndex(closes).tz_localize("UTC")
    return TradeLog.from_frame(pd.DataFrame({
        "open_time": close - pd.Timedelta(minutes=30), "close_time": close, "pnl": pnl,
        "units": units or [1.0] * n,
        "risk": risk or [np.nan] * n, "mae": mae or [np.nan] * n,
    }))


def test_tradelog_rejects_naive_times_and_bad_units():
    df = pd.DataFrame({"open_time": pd.to_datetime(["2024-01-01"]),
                       "close_time": pd.to_datetime(["2024-01-01"]), "pnl": [1.0]})
    with pytest.raises(TradeLogError, match="timezone"):
        TradeLog.from_frame(df)
    with pytest.raises(TradeLogError, match="units"):
        _log([1.0], units=[0.0])


def test_adverse_models():
    log = _log([100.0, -50.0], risk=[80.0, 80.0], mae=[30.0, 20.0])
    np.testing.assert_allclose(adverse_amount(log, IntratradeModel.REALIZED), [0, 50])
    np.testing.assert_allclose(adverse_amount(log, IntratradeModel.MAE), [30, 50])
    np.testing.assert_allclose(adverse_amount(log, IntratradeModel.STOP_BOUND), [80, 80])
    assert build_daybook(log, SessionCalendar()).intratrade is IntratradeModel.MAE
    with pytest.raises(ValueError, match="mae"):
        adverse_amount(_log([1.0]), IntratradeModel.MAE)


def test_discrete_sizing_rounds_and_caps():
    s = SizingModel(unit_step=1, max_units=3)
    np.testing.assert_allclose(s.scale_units(np.array([1, 1, 3, 4]), 0.4), [0, 0, 1, 2])
    np.testing.assert_allclose(s.scale_units(np.array([1, 3]), 0.5), [1, 2])  # half rounds up
    np.testing.assert_allclose(s.scale_units(np.array([2]), 2.0), [3])        # capped
    np.testing.assert_allclose(SizingModel().scale_units(np.array([2.0]), 0.25), [0.5])


def test_daybook_groups_by_firm_day_and_scales_per_unit():
    log = _log([100.0, -40.0, 30.0], units=[2.0, 1.0, 1.0],
               closes=["2024-01-15 20:00", "2024-01-15 22:30", "2024-01-16 15:00"])
    cal = SessionCalendar(timezone="America/Chicago", rollover=dt.time(17, 0))
    book = build_daybook(log, cal)
    # 22:30 UTC = 16:30 CST -> still the 15th; nothing crosses 17:00 CT
    assert [str(d) for d in book.days] == ["2024-01-15", "2024-01-16"]
    np.testing.assert_allclose(book.daily_pnl(), [60.0, 30.0])
    pnl, adv, units = book.scaled(0.5, SizingModel(unit_step=1))
    np.testing.assert_allclose(units, [1, 1, 1])
    np.testing.assert_allclose(pnl, [50.0, -40.0, 30.0])
    np.testing.assert_allclose(adv, [0.0, 40.0, 0.0])


# ─── stats (golden, hand computed) ───────────────────────────────────────────


def test_stats_golden():
    pnl = np.array([100, -50, -50, 200, -300, 100], dtype=float)
    eq = stats.equity_curve(pnl, 1000)
    np.testing.assert_allclose(eq, [1000, 1100, 1050, 1000, 1200, 900, 1000])
    assert stats.max_drawdown(eq) == 300
    assert stats.max_drawdown_pct(eq) == pytest.approx(0.25)
    assert stats.avg_drawdown(eq) == pytest.approx((50 + 100 + 300 + 200) / 4)
    assert stats.profit_factor(pnl) == pytest.approx(400 / 400)
    assert stats.win_rate(pnl) == pytest.approx(0.5)
    assert stats.longest_losing_streak(pnl) == 2
    assert stats.var(pnl, 0.8) == 50.0          # lower 20% quantile of 6 values = -50
    assert stats.cvar(pnl, 0.8) == pytest.approx((300 + 50 + 50) / 3)
    s = stats.summarize(pnl)
    assert (s.total, s.best, s.worst, s.avg_loss) == (0.0, 200.0, -300.0, pytest.approx(-400 / 3))


# ─── presets ─────────────────────────────────────────────────────────────────


def test_bundled_presets_load_with_provenance():
    ids = {p.id for p in list_presets()}
    assert {"ftmo-2step", "ftmo-1step", "ftmo-1step-bestday", "topstep-50k"} <= ids
    for p in list_presets():
        assert p.source_urls and p.verified_at


def test_pct_presets_rescale_with_balance():
    spec = load_preset("ftmo-2step").with_balance(25_000)
    floors = spec.phases[0].floors
    assert [f.value(spec.initial_balance) for f in floors] == [1250, 2500]
    assert spec.phases[1].profit_target.value(spec.initial_balance) == 1250
