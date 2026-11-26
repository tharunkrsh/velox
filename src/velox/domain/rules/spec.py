"""Declarative prop-firm rule specifications.

Rules are data: a preset YAML file compiles to a :class:`RulesetSpec`. Each rule
is a *pure function of the account state* (see :class:`Account`), which keeps the
reference engine and the vectorised engine trivially consistent.

Two rule families:

* **Floor rules** define a balance/equity level that must not be breached
  (static max loss, EOD trailing, intraday trailing, daily loss).
* **Pass conditions** must all hold for a phase to pass
  (profit target, minimum trading days, best-day consistency).

Amounts are either absolute (``amount``) or a fraction of the initial balance
(``pct``), so percentage-based presets work for any account size.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from velox.domain.calendar import SessionCalendar

# ─── Account state (read by rules, mutated only by the engine) ────────────────


@dataclass
class Account:
    initial: float
    balance: float
    day_start_balance: float
    eod_peak: float            # max(initial, every completed day's closing balance)
    equity_peak: float         # max(initial, every balance seen after a trade close)
    day_pnl: float = 0.0
    traded_today: bool = False
    trading_days: int = 0      # completed days with >= 1 executed trade
    closed_day_pnls: list[float] = field(default_factory=list)

    @classmethod
    def fresh(cls, initial: float) -> Account:
        return cls(initial, initial, initial, initial, initial)

    @property
    def profit(self) -> float:
        return self.balance - self.initial

    @property
    def day_pnls_incl_today(self) -> list[float]:
        return [*self.closed_day_pnls, self.day_pnl] if self.traded_today else self.closed_day_pnls

    @property
    def trading_days_incl_today(self) -> int:
        return self.trading_days + int(self.traded_today)


class FailReason(StrEnum):
    DAILY_LOSS = "daily_loss"
    MAX_LOSS = "max_loss"


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

    def value(self, initial: float) -> float:
        return self.amount if self.amount is not None else self.pct * initial


# ─── Floor rules ──────────────────────────────────────────────────────────────


class _Floor(_Amount):
    includes_floating: bool = True  # check equity while trades are open, not just balance
    breach_on_touch: bool = True    # equity == floor counts as a breach

    reason: FailReason = FailReason.MAX_LOSS
    effect: Literal["fail", "halt_day"] = "fail"

    def floor(self, a: Account) -> float:
        raise NotImplementedError

    def breached(self, value: float, a: Account) -> bool:
        f = self.floor(a)
        return value <= f if self.breach_on_touch else value < f


class StaticMaxLoss(_Floor):
    type: Literal["static_max_loss"] = "static_max_loss"

    def floor(self, a: Account) -> float:
        return a.initial - self.value(a.initial)


class EodTrailingMaxLoss(_Floor):
    """Floor trails the highest end-of-day balance.

    ``lock_offset``: once the floor reaches ``initial + lock_offset`` it stops
    trailing (Topstep locks at the starting balance, i.e. offset 0). None = never locks.
    """

    type: Literal["eod_trailing_max_loss"] = "eod_trailing_max_loss"
    lock_offset: float | None = 0.0

    def floor(self, a: Account) -> float:
        f = a.eod_peak - self.value(a.initial)
        return f if self.lock_offset is None else min(f, a.initial + self.lock_offset)


class IntradayTrailingMaxLoss(_Floor):
    """Floor trails the highest balance seen after any trade close.

    Without MFE data the true intraday equity peak is unknown; using closed
    balances is optimistic (it under-estimates the peak, so the floor is lower).
    """

    type: Literal["intraday_trailing_max_loss"] = "intraday_trailing_max_loss"
    lock_offset: float | None = 0.0

    def floor(self, a: Account) -> float:
        f = a.equity_peak - self.value(a.initial)
        return f if self.lock_offset is None else min(f, a.initial + self.lock_offset)


class DailyLossLimit(_Floor):
    """Floor = start-of-day balance minus the limit. Resets every trading day.

    ``effect="halt_day"`` models firms that liquidate and lock the account for
    the rest of the day instead of failing the evaluation.
    """

    type: Literal["daily_loss_limit"] = "daily_loss_limit"
    reason: FailReason = FailReason.DAILY_LOSS

    def floor(self, a: Account) -> float:
        return a.day_start_balance - self.value(a.initial)


# ─── Pass conditions ──────────────────────────────────────────────────────────


class ProfitTarget(_Amount):
    type: Literal["profit_target"] = "profit_target"

    def met(self, a: Account) -> bool:
        return a.profit >= self.value(a.initial) - 1e-9


class MinTradingDays(_Base):
    type: Literal["min_trading_days"] = "min_trading_days"
    days: int = Field(ge=1)

    def met(self, a: Account) -> bool:
        return a.trading_days_incl_today >= self.days


class BestDayConsistency(_Base):
    """Best single day must not exceed ``max_share`` of the denominator.

    denominator  total_profit          current balance - initial (Topstep)
                 positive_days_profit  sum of all winning days (FTMO 1-Step Best Day)
    Failing this never fails the account; it delays the pass until satisfied.
    """

    type: Literal["best_day_consistency"] = "best_day_consistency"
    max_share: float = Field(gt=0, le=1)
    denominator: Literal["total_profit", "positive_days_profit"] = "total_profit"
    strict: bool = False

    def met(self, a: Account) -> bool:
        pnls = a.day_pnls_incl_today
        best = max(pnls, default=0.0)
        if best <= 0:
            return True
        denom = a.profit if self.denominator == "total_profit" else sum(p for p in pnls if p > 0)
        if denom <= 0:
            return False
        share = best / denom
        return share < self.max_share if self.strict else share <= self.max_share + 1e-12


FloorRule = Annotated[
    StaticMaxLoss | EodTrailingMaxLoss | IntradayTrailingMaxLoss | DailyLossLimit,
    Field(discriminator="type"),
]
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


class RulesetSpec(_Base):
    id: str
    name: str
    version: str
    source_urls: list[str]
    verified_at: dt.date
    needs_verification: list[str] = Field(default_factory=list)  # open questions on the rules
    notes: list[str] = Field(default_factory=list)

    initial_balance: float = Field(gt=0)
    list_fee: float | None = Field(default=None, ge=0)
    calendar: SessionCalendar
    phases: list[PhaseSpec] = Field(min_length=1)

    def with_balance(self, initial_balance: float) -> RulesetSpec:
        """Same rules at a different account size (pct amounts rescale; absolute don't)."""
        return self.model_copy(update={"initial_balance": initial_balance})
