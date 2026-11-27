"""Run every hand-computed golden scenario through both rule engines.

The vectorised engine is checked against the hand-computed expectations directly
(not only against the reference engine), so logic shared by the two engines
still has an independent oracle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from velox.domain.daybook import DayBook
from velox.domain.rules.reference import run_path
from velox.domain.rules.spec import RulesetSpec
from velox.domain.sizing import IntratradeModel
from velox.presets import load_preset
from velox.sim.engine import FUNDED_OUTCOMES, OUTCOMES, RULE_TYPES, simulate

DOC = yaml.safe_load((Path(__file__).parent / "rules" / "scenarios.yaml").read_text("utf-8"))
RULESETS = {k: RulesetSpec.model_validate(v) for k, v in DOC["rulesets"].items()}


def _spec(sc) -> RulesetSpec:
    return load_preset(sc["preset"]) if "preset" in sc else RULESETS[sc["ruleset"]]


def _days(raw):
    out = []
    for day in raw:
        pnl = np.array([t[0] for t in day], dtype=float)
        adv = np.array([t[1] if len(t) > 1 else max(0.0, -t[0]) for t in day], dtype=float)
        out.append((pnl, adv, np.ones(len(day))))
    return out


@pytest.mark.parametrize("sc", DOC["scenarios"], ids=lambda s: s["name"])
def test_scenario(sc):
    r = run_path(_spec(sc), _days(sc["days"]), max_days=sc.get("max_days"),
                 funded_days=sc.get("funded_days"), payout_cap=sc.get("payout_cap"))
    for field, expected in sc["expect"].items():
        assert getattr(r, field) == expected, f"{field}: got {getattr(r, field)!r}"


# Scenarios that end because the history runs out have no equivalent in the
# simulator (which always has enough days); every conclusive scenario must match.
CONCLUSIVE = [sc for sc in DOC["scenarios"]
              if "incomplete" not in (sc["expect"].get("outcome"),
                                      sc["expect"].get("funded_outcome"))]
LANE_FIELDS = {"outcome", "fail_rule", "phase_reached", "days_used", "consistency_delay_days",
               "halted_days", "funded_outcome", "funded_fail_rule", "funded_days_used",
               "n_payouts", "gross_payouts", "trader_payouts"}


def _lane_value(res, field):
    v = getattr(res, field)[0, 0]
    if field == "outcome":
        return str(OUTCOMES[v])
    if field == "funded_outcome":
        return None if v < 0 else str(FUNDED_OUTCOMES[v])
    if field in ("fail_rule", "funded_fail_rule"):
        return None if v < 0 else RULE_TYPES[v]
    return float(v) if field.endswith("payouts") and field != "n_payouts" else int(v)


@pytest.mark.parametrize("sc", CONCLUSIVE, ids=lambda s: s["name"])
def test_scenario_vectorised(sc):
    days = _days(sc["days"])
    counts = np.array([len(d[0]) for d in days] + [0])  # trailing empty day = padding
    book = DayBook(
        days=np.arange(len(counts)).astype("datetime64[D]"),
        offsets=np.concatenate([[0], np.cumsum(counts)]),
        units=np.ones(int(counts.sum())),
        pnl_per_unit=np.concatenate([d[0] for d in days]),
        adverse_per_unit=np.concatenate([d[1] for d in days]),
        intratrade=IntratradeModel.MAE,
    )
    max_eval = sc.get("max_days") or len(days) + 1
    funded = sc.get("funded_days") or 0
    horizon = max_eval + funded
    idx = np.minimum(np.arange(horizon), len(days))[None, :]  # history, then padding
    res = simulate(_spec(sc), book, idx, np.array([1.0]), max_eval_days=max_eval,
                   funded_days=funded, payout_cap=sc.get("payout_cap"))
    checked = [f for f in sc["expect"] if f in LANE_FIELDS]
    assert checked
    for field in checked:
        got = _lane_value(res, field)
        assert got == sc["expect"][field], f"{field}: got {got!r}"
