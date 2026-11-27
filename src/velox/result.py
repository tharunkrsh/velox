"""``SimulationResult``: the single canonical payload of a simulation run.

The CLI writes it as JSON, the HTML tearsheet is rendered from it, and the
Phase 3 API will return it unchanged, so every surface shows the same numbers.
It contains no timestamps or timings, so the same inputs and seed produce
byte-identical JSON.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from velox.domain.stats import PnlSummary
from velox.ingest.quality import DataQualityReport

SCHEMA_VERSION = "1.0"


class _M(BaseModel):
    model_config = ConfigDict(frozen=True)


class Prob(_M):
    """Point estimate with a 95% Wilson score interval."""

    p: float
    lo: float
    hi: float
    n: int


class MultiplierStats(_M):
    k: float
    # evaluation
    p_pass: Prob
    p_fail_daily_loss: Prob
    p_fail_max_loss: Prob
    p_timeout: Prob
    p_consistency_delay: Prob        # passed, but later than the day the target was first hit
    mean_delay_days: float | None    # among delayed passes
    days_to_pass_p50: float | None
    days_to_pass_p90: float | None
    fail_rules: dict[str, float]     # share of all paths failing on each rule type
    # funded stage
    p_payout: Prob                   # at least one payout within the funded horizon
    p_funded_breach: Prob            # passed the evaluation, then breached while funded
    mean_payouts: float              # mean number of payouts per path
    mean_net_payout: float           # E[trader payouts + fee refund], 0 for failed paths
    # economics
    ev: float                        # mean_net_payout - fee - p_pass * activation_fee
    ev_lo: float                     # 95% CI (normal approximation)
    ev_hi: float
    breakeven_funded_payout: float | None  # fee / p_pass + activation_fee


class StressPoint(_M):
    k: float
    p_pass: Prob
    ev: float


class StressCase(_M):
    haircut: float                   # fraction of mean trade P&L removed
    grid: list[StressPoint]
    recommended_k: float
    at_base_recommendation: StressPoint
    at_current: StressPoint


class Recommendation(_M):
    k: float
    k_current: float
    ev: float
    ev_current: float
    p_pass: Prob
    p_pass_current: Prob
    risk_per_trade_current: float | None   # median historical risk per trade
    risk_per_trade_recommended: float | None
    rationale: list[str]


class FanChart(_M):
    k: float
    label: str
    days: list[int]
    quantiles: dict[str, list[float]]  # "p5", "p25", "p50", "p75", "p95" per day
    samples: list[list[float]]         # a few individual balance paths


class RulesetInfo(_M):
    id: str
    name: str
    version: str
    verified_at: str
    source_urls: list[str]
    needs_verification: list[str]
    notes: list[str]
    initial_balance: float
    phases: list[str]
    phase_targets: list[float]       # profit target of each phase (amount above start)
    funded: str | None
    funded_start_balance: float | None
    profit_split: float | None


class RunConfig(_M):
    n_paths: int
    seed: int
    multipliers: list[float]
    resampler: str
    block_len: float
    max_eval_days: int
    funded_days: int
    payout_cap: float | None
    challenge_fee: float
    activation_fee: float
    fee_refund_on_first_payout: bool
    haircuts: list[float]
    unit_step: float | None
    max_units: float | None
    intratrade: str


class InputSummary(_M):
    quality: DataQualityReport
    trades: PnlSummary
    days: PnlSummary


class SimulationResult(_M):
    schema_version: str = SCHEMA_VERSION
    velox_version: str
    ruleset: RulesetInfo
    config: RunConfig
    inputs: InputSummary
    grid: list[MultiplierStats]
    recommendation: Recommendation
    stress: list[StressCase]
    fans: list[FanChart]
    warnings: list[str]
    disclaimers: list[str]
