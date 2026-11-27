"""Declarative prop-firm rule specifications.

Rules are data: a preset YAML file compiles to a :class:`RulesetSpec`. Each rule
is a *pure function of the account state*, written with NumPy operations so the
same method works on a scalar :class:`Account` (reference engine) and on an
array-valued lane view (vectorised Monte Carlo engine). One implementation of
the rule maths serves both engines.

Rule families:

* **Floor rules**: a balance/equity level that must not be breached
  (static max loss, EOD trailing, intraday trailing, daily loss).
* **Pass conditions**: must all hold for an evaluation phase to pass
  (profit target, minimum trading days, best-day consistency).
* **Funded stage**: an optional final stage with its own floors and a payout
  policy, simulated for a fixed horizon to value a passed evaluation.

Amounts are absolute (``amount``) or a fraction of the account size (``pct``).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, fields
from typing import Annotated, ClassVar, Literal, get_args

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from velox.domain.calendar import SessionCalendar
from velox.domain.outcomes import FailReason

# ─── Account state (read by rules, mutated only by the engines) ───────────────


@dataclass
class Account:
    """Account state for one stage: scalars for one path (reference engine), or
    one NumPy array per field across lanes (vectorised engine, see :meth:`lanes`).
    """

    initial: float             # stage starting balance (Topstep XFA starts at 0)
    basis: float               # account size that pct amounts refer to
    balance: float
    day_start_balance: float
    eod_peak: float            # max(initial, completed days' closing balances)
    equity_peak: float         # max(initial, balances after each trade close)
    day_pnl: float = 0.0
    traded_today: bool = False
    trading_days: int = 0      # completed days with >= 1 executed trade
    best_closed: float = -np.inf   # best completed day P&L
    pos_closed: float = 0.0        # sum of positive completed days
    payouts: int = 0               # funded stage: payouts taken so far

    @classmethod
    def fresh(cls, initial: float, basis: float) -> Account:
        return cls(initial, basis, initial, initial, initial, initial)

    # ── lane form: every field is an array over lanes ──

    @classmethod
    def lanes(cls, n: int, initial: float, basis: float) -> Account:
        """``n`` fresh accounts as one array-valued Account."""
        one = cls.fresh(initial, basis)
        return cls(**{f: np.full(n, getattr(one, f)) for f in _ACCOUNT_FIELDS})

    def take(self, idx: np.ndarray, **override) -> Account:
        """Copy of the lanes at ``idx``; ``override`` supplies already-sliced fields."""
        return Account(**{f: getattr(self, f)[idx] for f in _ACCOUNT_FIELDS if f not in override},
                       **override)

    def put(self, idx: np.ndarray, other: Account) -> None:
        """Write ``other`` (lanes or a scalar account) into the lanes at ``idx``."""
        for f in _ACCOUNT_FIELDS:
            getattr(self, f)[idx] = getattr(other, f)

    def reset(self, idx: np.ndarray, initial: float) -> None:
        """Start the lanes at ``idx`` on a fresh account (new stage)."""
        self.put(idx, Account.fresh(initial, self.basis[idx]))

    @property
    def profit(self):
        return self.balance - self.initial

    @property
    def best_day_incl_today(self):
        return np.where(self.traded_today, np.maximum(self.best_closed, self.day_pnl),
                        self.best_closed)

    @property
    def positive_days_profit_incl_today(self):
        return self.pos_closed + np.where(self.traded_today, np.maximum(self.day_pnl, 0.0), 0.0)

    @property
    def trading_days_incl_today(self):
        return self.trading_days + np.asarray(self.traded_today, dtype=int)


_ACCOUNT_FIELDS = tuple(f.name for f in fields(Account))


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Amount(_Base):
    amount: float | None = Field(default=None, gt=0)
    pct: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def _one_of(self) -> _Amount:
        if (self.amount is None) == (self.pct is None):
            raise ValueError("give exactly one of 'amount' or 'pct'")
        return self

    def value(self, basis):
        return self.amount if self.amount is not None else self.pct * basis


# ─── Floor rules ──────────────────────────────────────────────────────────────


class _Floor(_Amount):
    # the floor level can change between trades of one day (not just at day boundaries)
    moves_intraday: ClassVar[bool] = False

    includes_floating: bool = True  # check equity while trades are open, not just balance
    breach_on_touch: bool = True    # equity == floor counts as a breach

    reason: FailReason = FailReason.MAX_LOSS
    effect: Literal["fail", "halt_day"] = "fail"

    def floor(self, a):
        raise NotImplementedError

    def breached(self, value, a):
        f = self.floor(a)
        return value <= f if self.breach_on_touch else value < f


class StaticMaxLoss(_Floor):
    type: Literal["static_max_loss"] = "static_max_loss"

    def floor(self, a):
        return a.initial - self.value(a.basis)


class _Trailing(_Floor):
    """``lock_offset``: the floor stops trailing at ``initial + lock_offset``
    (Topstep locks at the starting balance: 0). None = never locks.
    ``post_payout_lock_offset``: after the first payout the floor is fixed at
    ``initial + offset`` (Topstep XFA: MLL resets to $0 permanently)."""

    lock_offset: float | None = 0.0
    post_payout_lock_offset: float | None = None

    def _peak(self, a):
        raise NotImplementedError

    def floor(self, a):
        f = self._peak(a) - self.value(a.basis)
        if self.lock_offset is not None:
            f = np.minimum(f, a.initial + self.lock_offset)
        if self.post_payout_lock_offset is not None:
            f = np.where(np.asarray(a.payouts) > 0, a.initial + self.post_payout_lock_offset, f)
        return f


class EodTrailingMaxLoss(_Trailing):
    """Floor trails the highest end-of-day balance."""

    type: Literal["eod_trailing_max_loss"] = "eod_trailing_max_loss"

    def _peak(self, a):
        return a.eod_peak


class IntradayTrailingMaxLoss(_Trailing):
    """Floor trails the highest balance seen after any trade close.

    Without MFE data the true intraday equity peak is unknown; using closed
    balances is optimistic (it under-estimates the peak, so the floor is lower).
    """

    type: Literal["intraday_trailing_max_loss"] = "intraday_trailing_max_loss"
    moves_intraday: ClassVar[bool] = True

    def _peak(self, a):
        return a.equity_peak


class DailyLossLimit(_Floor):
    """Floor = start-of-day balance minus the limit. Resets every trading day.

    ``effect="halt_day"`` models firms that liquidate and lock the account for
    the rest of the day instead of failing the evaluation.
    """

    type: Literal["daily_loss_limit"] = "daily_loss_limit"
    reason: FailReason = FailReason.DAILY_LOSS

    def floor(self, a):
        return a.day_start_balance - self.value(a.basis)


# ─── Pass conditions ──────────────────────────────────────────────────────────


class ProfitTarget(_Amount):
    type: Literal["profit_target"] = "profit_target"

    def met(self, a):
        return a.profit >= self.value(a.basis) - 1e-9


class MinTradingDays(_Base):
    type: Literal["min_trading_days"] = "min_trading_days"
    days: int = Field(ge=1)

    def met(self, a):
        return a.trading_days_incl_today >= self.days


class BestDayConsistency(_Base):
    """Best single day must not exceed ``max_share`` of the denominator.

    denominator  total_profit          current balance - stage start, losing days
                                       included (Topstep; FTMO 1-Step Consistency rule)
    positive_days_profit  sum of winning days only (FTMO legacy Best Day Rule)
    Failing this never fails the account; it delays the pass until satisfied.
    """

    type: Literal["best_day_consistency"] = "best_day_consistency"
    max_share: float = Field(gt=0, le=1)
    denominator: Literal["total_profit", "positive_days_profit"] = "total_profit"
    strict: bool = False

    def met(self, a):
        best = a.best_day_incl_today
        denom = a.profit if self.denominator == "total_profit" else \
            a.positive_days_profit_incl_today
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(denom > 0, best / np.where(denom > 0, denom, 1.0), np.inf)
        ok = share < self.max_share if self.strict else share <= self.max_share + 1e-12
        return (best <= 0) | ok


FloorRule = Annotated[
    StaticMaxLoss | EodTrailingMaxLoss | IntradayTrailingMaxLoss | DailyLossLimit,
    Field(discriminator="type"),
]
# floor ``type`` values in union order; the vectorised engine stores fail rules as indices
FLOOR_TYPES: tuple[str, ...] = tuple(
    cls.model_fields["type"].default for cls in get_args(get_args(FloorRule)[0]))
FLOOR_CODE = {t: i for i, t in enumerate(FLOOR_TYPES)}

PassRule = Annotated[
    ProfitTarget | MinTradingDays | BestDayConsistency, Field(discriminator="type")
]


class PhaseSpec(_Base):
    name: str
    floors: list[FloorRule]
    pass_conditions: list[PassRule]
    stop_on_target: bool = True  # trader stops for the day once all pass conditions hold

    @model_validator(mode="after")
    def _has_target(self) -> PhaseSpec:
        if not any(isinstance(r, ProfitTarget) for r in self.pass_conditions):
            raise ValueError(f"phase '{self.name}' has no profit_target")
        return self

    @property
    def profit_target(self) -> ProfitTarget:
        return next(r for r in self.pass_conditions if isinstance(r, ProfitTarget))

    def pass_ok(self, a):
        ok = True
        for r in self.pass_conditions:
            ok = ok & r.met(a)
        return ok

    def target_met(self, a):
        return self.profit_target.met(a)


# ─── Funded stage ─────────────────────────────────────────────────────────────


class PayoutPolicy(_Base):
    """When and how much a funded trader can withdraw (checked at each day end).

    Eligible when, since the stage start or the last payout, at least
    ``min_days_between`` trading days and ``min_winning_days`` days with
    P&L >= ``winning_day_threshold`` have passed. The gross request is
    ``max_share`` of the profit above the stage start balance, capped by
    ``max_amount`` (and the run's payout cap); the trader receives ``split`` of it.
    """

    split: float = Field(gt=0, le=1)
    min_days_between: int = Field(default=0, ge=0)
    min_winning_days: int = Field(default=0, ge=0)
    winning_day_threshold: float = 0.0
    max_share: float = Field(default=1.0, gt=0, le=1)
    max_amount: float | None = Field(default=None, gt=0)
    min_amount: float = Field(default=0.0, ge=0)
    refund_challenge_fee: bool = False  # fee refunded with the first payout (FTMO 2-Step)
    reset_peaks_on_payout: bool = True  # trailing peaks restart from the post-payout balance

    def gross_request(self, profit, cap: float | None = None):
        amount = self.max_share * np.maximum(profit, 0.0)
        limit = min(x for x in (self.max_amount, cap, np.inf) if x is not None)
        return np.minimum(amount, limit)


class FundedSpec(_Base):
    name: str
    initial_balance: float | None = None  # None = the evaluation account size
    floors: list[FloorRule]
    payout: PayoutPolicy
    activation_fee: float = Field(default=0.0, ge=0)


class RulesetSpec(_Base):
    id: str
    name: str
    version: str
    source_urls: list[str]
    verified_at: dt.date
    needs_verification: list[str] = Field(default_factory=list)  # open questions on the rules
    notes: list[str] = Field(default_factory=list)

    initial_balance: float = Field(gt=0)
    calendar: SessionCalendar
    phases: list[PhaseSpec] = Field(min_length=1)
    funded: FundedSpec | None = None

    def with_balance(self, initial_balance: float) -> RulesetSpec:
        """Same rules at a different account size (pct amounts rescale; absolute don't)."""
        return self.model_copy(update={"initial_balance": initial_balance})

    @property
    def funded_start(self) -> float:
        if self.funded is None or self.funded.initial_balance is None:
            return self.initial_balance
        return self.funded.initial_balance
