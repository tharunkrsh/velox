from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from velox.app.cli import app
from velox.domain.calendar import SessionCalendar
from velox.domain.trades import TradeLogError
from velox.ingest import load_trades
from velox.ingest.quality import assess

FIX = Path(__file__).parent.parent / "fixtures"


def test_generic_canonical_csv_with_tz():
    log = load_trades(FIX / "topstep_sample.csv", timezone="America/Chicago")
    assert len(log) == 8
    assert str(log.close_time.tz) == "UTC"
    assert log.close_time[0].hour == 15  # 09:05 CST -> 15:05 UTC
    np.testing.assert_allclose(log.pnl[:2], [600.0, 700.0])


def test_naive_timestamps_require_timezone():
    with pytest.raises(TradeLogError, match="timezone"):
        load_trades(FIX / "topstep_sample.csv")


def test_mt5_positions_layout():
    log = load_trades(FIX / "mt5_positions.csv", fmt="mt5", timezone="Europe/Prague")
    assert len(log) == 3
    np.testing.assert_allclose(log.pnl, [120.0 - 7.0, -80.0 - 7.0 - 1.5, 50.0 - 3.5])
    np.testing.assert_allclose(log.units, [1.0, 1.0, 0.5])
    assert list(log.symbol) == ["EURUSD", "XAUUSD", "EURUSD"]


def test_quality_flags_overlap_and_small_sample():
    log = load_trades(FIX / "mt5_positions.csv", fmt="mt5", timezone="Europe/Prague")
    q = assess(log, SessionCalendar(timezone="Europe/Prague"))
    assert q.overlapping_trades == 1
    assert any("Small sample" in w for w in q.warnings)
    assert any("No MAE" in w for w in q.warnings)


runner = CliRunner()


def test_cli_replay_matches_hand_computation():
    # Hand computation (Topstep 50K, 17:00 CT days): see fixture header comment.
    res = runner.invoke(app, ["replay", str(FIX / "topstep_sample.csv"), "-p", "topstep-50k",
                              "--tz", "America/Chicago"])
    assert res.exit_code == 0, res.output
    lines = [ln.strip() for ln in res.output.splitlines() if "attempt" in ln]
    assert lines[0].endswith("PASSED all phases in 4 trading days")
    assert "attempt 2" in lines[1] and "FAILED" in lines[1] and "eod_trailing_max_loss" in lines[1]


def test_cli_inspect_and_presets():
    res = runner.invoke(app, ["inspect", str(FIX / "topstep_sample.csv"), "-p", "topstep-50k",
                              "--tz", "America/Chicago"])
    assert res.exit_code == 0, res.output
    assert "8 trades over 7 trading days" in res.output
    assert "topstep-50k" in runner.invoke(app, ["presets"]).output


def test_cli_reports_missing_tz_cleanly():
    res = runner.invoke(app, ["inspect", str(FIX / "topstep_sample.csv"), "-p", "topstep-50k"])
    assert res.exit_code == 2
