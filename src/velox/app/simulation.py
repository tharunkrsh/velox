"""Run a full simulation and assemble the canonical :class:`SimulationResult`.

This is the one orchestration entry point: the CLI calls it today and the
Phase 3 API will call it unchanged. Rendering lives in :mod:`velox.report`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

import velox
from velox.domain.daybook import DayBook, build_daybook
from velox.domain.rules.spec import RulesetSpec
from velox.domain.sizing import IntratradeModel, SizingModel
from velox.domain.stats import summarize
from velox.domain.trades import TradeLog
from velox.ingest.quality import assess
from velox.result import (
    FanChart,
    InputSummary,
    MultiplierStats,
    Recommendation,
    RulesetInfo,
    RunConfig,
    SimulationResult,
    StressCase,
)
from velox.sim.engine import LaneResults, simulate
from velox.sim.optimizer import best_index, grid_stats, stress_point
from velox.sim.regime import InsufficientData, fit_regimes
from velox.sim.resample import regime_bootstrap, stationary_bootstrap

DISCLAIMERS = [
    "Not affiliated with, endorsed by or sponsored by any prop firm. Firm names identify the "
    "public rule sets that were modelled.",
    "Not financial advice. Results resample YOUR past trades; past performance does not "
    "guarantee future results, and edges decay.",
    "Firm rules change frequently. Check the preset's verification date and open questions "
    "against your own account terms.",
]


class SimulationRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    challenge_fee: float = Field(ge=0)
    activation_fee: float | None = Field(default=None, ge=0)  # None = preset value
    n_paths: int = Field(default=10_000, ge=100)
    seed: int = 7
    k_min: float = Field(default=0.25, gt=0)
    k_max: float = Field(default=2.0, gt=0)
    k_step: float = Field(default=0.05, gt=0)
    resampler: Literal["stationary", "regime"] = "stationary"
    block_len: float = Field(default=4.0, ge=1)
    max_eval_days: int = Field(default=60, ge=1)
    funded_days: int = Field(default=60, ge=0)
    payout_cap: float | None = Field(default=None, gt=0)
    haircuts: list[float] = Field(default_factory=lambda: [0.25, 0.5])
    stress_step: float = Field(default=0.25, gt=0)
    unit_step: float | None = Field(default=None, gt=0)
    max_units: float | None = Field(default=None, gt=0)
    intratrade: IntratradeModel = IntratradeModel.AUTO
    n_samples: int = Field(default=8, ge=0)

    def multipliers(self) -> np.ndarray:
        n = int(round((self.k_max - self.k_min) / self.k_step)) + 1
        ks = np.round(self.k_min + self.k_step * np.arange(n), 6)
        return np.unique(np.append(ks, 1.0))  # always include the current size


def _day_index(book: DayBook, req: SimulationRequest, horizon: int,
               rng: np.random.Generator, warnings: list[str]) -> tuple[np.ndarray, str]:
    if req.resampler == "regime":
        try:
            model = fit_regimes(book.daily_pnl(), seed=req.seed)
            return regime_bootstrap(model, req.n_paths, horizon, rng=rng), "regime"
        except (InsufficientData, ImportError) as e:
            warnings.append(f"Regime-aware resampling unavailable ({e}); used the stationary "
                            "block bootstrap instead.")
    return (stationary_bootstrap(book.n_days, req.n_paths, horizon, block_len=req.block_len,
                                 rng=rng), "stationary")


def _fan(balances: np.ndarray, initial: float, k: float, label: str, n_samples: int) -> FanChart:
    filled = pd.DataFrame(balances).ffill(axis=1).to_numpy()
    filled = np.column_stack([np.full(len(filled), initial), filled])
    qs = {f"p{q}": [round(float(v), 2) for v in np.quantile(filled, q / 100, axis=0)]
          for q in (5, 25, 50, 75, 95)}
    return FanChart(
        k=k, label=label, days=list(range(filled.shape[1])), quantiles=qs,
        samples=[[round(float(v), 2) for v in row] for row in filled[:n_samples]],
    )


def _ruleset_info(spec: RulesetSpec) -> RulesetInfo:
    f = spec.funded
    return RulesetInfo(
        id=spec.id, name=spec.name, version=spec.version, verified_at=str(spec.verified_at),
        source_urls=spec.source_urls, needs_verification=spec.needs_verification,
        notes=spec.notes, initial_balance=spec.initial_balance,
        phases=[p.name for p in spec.phases],
        phase_targets=[float(p.profit_target.value(spec.initial_balance)) for p in spec.phases],
        funded=f.name if f else None,
        funded_start_balance=spec.funded_start if f else None,
        profit_split=f.payout.split if f else None,
    )


def _economics(spec: RulesetSpec, req: SimulationRequest) -> tuple[float, bool]:
    """(activation fee, challenge fee refunded with the first payout?)"""
    f = spec.funded
    activation_fee = (req.activation_fee if req.activation_fee is not None
                      else (f.activation_fee if f else 0.0))
    return activation_fee, bool(f and f.payout.refund_challenge_fee)


Simulate = Callable[..., LaneResults]  # (book, multipliers, **simulate_kw) -> LaneResults


def _stress(sim: Simulate, book: DayBook, ks: np.ndarray, cur: MultiplierStats,
            rec: MultiplierStats, req: SimulationRequest, econ: dict) -> list[StressCase]:
    """Edge-haircut stress on a coarser grid (always including current + recommended)."""
    stride = max(1, int(round(req.stress_step / req.k_step)))
    coarse = np.unique(np.concatenate([ks[::stride], [cur.k, rec.k]]))
    stress = []
    for h in req.haircuts:
        sgrid = grid_stats(sim(book.haircut(h), coarse), **econ)
        by_k = {g.k: g for g in sgrid}
        stress.append(StressCase(
            haircut=h, grid=[stress_point(g) for g in sgrid],
            recommended_k=sgrid[best_index(sgrid)].k,
            at_base_recommendation=stress_point(by_k[rec.k]),
            at_current=stress_point(by_k[cur.k]),
        ))
    return stress


def _fans(sim: Simulate, book: DayBook, spec: RulesetSpec, cur: MultiplierStats,
          rec: MultiplierStats, n_samples: int) -> list[FanChart]:
    """Replay the same paths at current and recommended size, recording EOD balances."""
    fan_ks = np.array([cur.k] if rec.k == cur.k else [cur.k, rec.k])
    fres = sim(book, fan_ks, record_multipliers=list(range(len(fan_ks))))
    return [_fan(fres.eval_balances[j], spec.initial_balance, float(k),
                 "current size" if j == 0 else "recommended size", n_samples)
            for j, k in enumerate(fan_ks)]


def _rationale(cur: MultiplierStats, rec: MultiplierStats, stress: list[StressCase]) -> list[str]:
    lines = [
        f"x{rec.k:g} maximises expected value per attempt: {rec.ev:+,.0f} "
        f"(95% CI {rec.ev_lo:+,.0f} to {rec.ev_hi:+,.0f}) vs {cur.ev:+,.0f} at your current size.",
        f"Pass probability {rec.p_pass.p:.1%} (95% CI {rec.p_pass.lo:.1%}-{rec.p_pass.hi:.1%}) "
        f"vs {cur.p_pass.p:.1%} now.",
    ]
    for s in stress:
        lines.append(
            f"If {s.haircut:.0%} of your edge disappears, x{rec.k:g} gives EV "
            f"{s.at_base_recommendation.ev:+,.0f} and pass {s.at_base_recommendation.p_pass.p:.1%};"
            f" the best size would be x{s.recommended_k:g}.")
    return lines


def run_simulation(log: TradeLog, spec: RulesetSpec, req: SimulationRequest) -> SimulationResult:
    warnings: list[str] = []
    quality = assess(log, spec.calendar, req.intratrade)
    book = build_daybook(log, spec.calendar, req.intratrade)
    funded_days = req.funded_days if spec.funded is not None else 0
    if spec.funded is None and req.funded_days:
        warnings.append("This preset has no funded stage; EV counts no payouts.")
    activation_fee, refund = _economics(spec, req)
    econ = dict(fee=req.challenge_fee, activation_fee=activation_fee, refund=refund)

    rng = np.random.default_rng(req.seed)
    day_idx, resampler = _day_index(book, req, req.max_eval_days + funded_days, rng, warnings)
    sizing = SizingModel(unit_step=req.unit_step, max_units=req.max_units)

    def sim(b: DayBook, ks: np.ndarray, **kw) -> LaneResults:
        return simulate(spec, b, day_idx, ks, sizing=sizing, max_eval_days=req.max_eval_days,
                        funded_days=funded_days, payout_cap=req.payout_cap, **kw)

    ks = req.multipliers()
    grid = grid_stats(sim(book, ks), **econ)
    cur = grid[int(np.flatnonzero(ks == 1.0)[0])]
    rec = grid[best_index(grid)]
    stress = _stress(sim, book, ks, cur, rec, req, econ)
    fans = _fans(sim, book, spec, cur, rec, req.n_samples)
    risk_cur = float(np.median(log.risk)) if log.has_risk else None

    return SimulationResult(
        velox_version=velox.__version__,
        ruleset=_ruleset_info(spec),
        config=RunConfig(
            n_paths=req.n_paths, seed=req.seed, multipliers=[float(k) for k in ks],
            resampler=resampler, block_len=req.block_len, max_eval_days=req.max_eval_days,
            funded_days=funded_days, payout_cap=req.payout_cap,
            challenge_fee=req.challenge_fee, activation_fee=activation_fee,
            fee_refund_on_first_payout=refund, haircuts=list(req.haircuts),
            unit_step=req.unit_step, max_units=req.max_units,
            intratrade=str(book.intratrade),
        ),
        inputs=InputSummary(quality=quality, trades=summarize(log.pnl),
                            days=summarize(book.daily_pnl())),
        grid=grid,
        recommendation=Recommendation(
            k=rec.k, k_current=cur.k, ev=rec.ev, ev_current=cur.ev,
            p_pass=rec.p_pass, p_pass_current=cur.p_pass,
            risk_per_trade_current=risk_cur,
            risk_per_trade_recommended=round(risk_cur * rec.k, 2) if risk_cur else None,
            rationale=_rationale(cur, rec, stress),
        ),
        stress=stress,
        fans=fans,
        warnings=[*quality.warnings, *warnings],
        disclaimers=DISCLAIMERS,
    )
