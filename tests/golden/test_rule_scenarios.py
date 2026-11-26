"""Run every hand-computed golden scenario through the reference rule engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from velox.domain.rules.reference import run_path
from velox.domain.rules.spec import RulesetSpec
from velox.presets import load_preset

DOC = yaml.safe_load((Path(__file__).parent / "rules" / "scenarios.yaml").read_text("utf-8"))
RULESETS = {k: RulesetSpec.model_validate(v) for k, v in DOC["rulesets"].items()}


def _days(raw):
    out = []
    for day in raw:
        pnl = np.array([t[0] for t in day], dtype=float)
        adv = np.array([t[1] if len(t) > 1 else max(0.0, -t[0]) for t in day], dtype=float)
        out.append((pnl, adv, np.ones(len(day))))
    return out


@pytest.mark.parametrize("sc", DOC["scenarios"], ids=lambda s: s["name"])
def test_scenario(sc):
    spec = load_preset(sc["preset"]) if "preset" in sc else RULESETS[sc["ruleset"]]
    r = run_path(spec, _days(sc["days"]), max_days=sc.get("max_days"),
                 funded_days=sc.get("funded_days"), payout_cap=sc.get("payout_cap"))
    for field, expected in sc["expect"].items():
        assert getattr(r, field) == expected, f"{field}: got {getattr(r, field)!r}"
