"""How a simulated path ends, shared by both rule engines.

The vectorised engine stores outcomes in integer lane arrays; the codes are the
enum member positions (``OUTCOME_CODE`` / ``FUNDED_CODE``), with ``NONE`` (-1)
for "not applicable". Floor-rule codes live next to the rules
(:data:`velox.domain.rules.spec.FLOOR_TYPES`).
"""

from __future__ import annotations

from enum import StrEnum


class FailReason(StrEnum):
    DAILY_LOSS = "daily_loss"
    MAX_LOSS = "max_loss"


class Outcome(StrEnum):
    PASSED = "passed"
    FAIL_DAILY_LOSS = "fail_daily_loss"
    FAIL_MAX_LOSS = "fail_max_loss"
    TIMEOUT = "timeout"        # hit the evaluation max-days cap
    INCOMPLETE = "incomplete"  # ran out of trade history (replay)


class FundedOutcome(StrEnum):
    BREACHED = "breached"      # a funded fail rule was hit
    SURVIVED = "survived"      # reached the funded horizon intact
    INCOMPLETE = "incomplete"  # ran out of trade history (replay)


FAIL_OUTCOME = {FailReason.DAILY_LOSS: Outcome.FAIL_DAILY_LOSS,
                FailReason.MAX_LOSS: Outcome.FAIL_MAX_LOSS}

NONE = -1
OUTCOMES = tuple(Outcome)
FUNDED_OUTCOMES = tuple(FundedOutcome)
OUTCOME_CODE = {o: i for i, o in enumerate(OUTCOMES)}
FUNDED_CODE = {o: i for i, o in enumerate(FUNDED_OUTCOMES)}
