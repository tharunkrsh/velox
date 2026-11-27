"""Position-size scaling and the intra-trade (floating loss) model.

Sizing is a per-trade *unit* transform, not a P&L multiply: a 0.5x multiplier on
a 1-contract futures trade rounds to 0 or 1 contracts, which matters a lot at
small sizes. P&L and adverse excursion are carried per unit and rescaled.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from velox.domain.trades import TradeLog


class IntratradeModel(StrEnum):
    """How deep equity is assumed to dip while a trade is open.

    realized    trough = realised loss only (optimistic lower bound on breaches)
    mae         trough = recorded max adverse excursion
    stop_bound  trough = full planned risk on every trade (pessimistic upper bound)
    auto        mae if the log has it, else realized
    """

    AUTO = "auto"
    REALIZED = "realized"
    MAE = "mae"
    STOP_BOUND = "stop_bound"


def resolve_intratrade(log: TradeLog, model: IntratradeModel) -> IntratradeModel:
    if model is IntratradeModel.AUTO:
        return IntratradeModel.MAE if log.has_mae else IntratradeModel.REALIZED
    if model is IntratradeModel.MAE and not log.has_mae:
        raise ValueError("intratrade model 'mae' needs an MAE value on every trade")
    if model is IntratradeModel.STOP_BOUND and not log.has_risk:
        raise ValueError("intratrade model 'stop_bound' needs a risk value on every trade")
    return model


def adverse_amount(log: TradeLog, model: IntratradeModel) -> np.ndarray:
    """Positive amount below the pre-trade balance that equity reaches while open.

    Always at least the realised loss, since equity must pass through it.
    """
    model = resolve_intratrade(log, model)
    realised_loss = np.maximum(0.0, -log.pnl)
    match model:
        case IntratradeModel.REALIZED:
            return realised_loss
        case IntratradeModel.MAE:
            return np.maximum(log.mae, realised_loss)
        case IntratradeModel.STOP_BOUND:
            return np.maximum(log.risk, realised_loss)
    raise AssertionError(model)


class SizingModel(BaseModel):
    """How a multiplier k maps historical units to simulated units.

    unit_step  None = continuous (pure linear scaling); else round to this step
               (1 for whole futures contracts, 0.01 for FX lots)
    max_units  optional per-trade cap (e.g. a firm's max contract limit)
    """

    model_config = ConfigDict(frozen=True)

    unit_step: float | None = Field(default=None, gt=0)
    max_units: float | None = Field(default=None, gt=0)

    def scale_units(self, units: np.ndarray, k: float) -> np.ndarray:
        scaled = np.asarray(units, dtype=float) * k
        if self.unit_step is not None:
            # round half up, robust to float noise (0.5 * 3 = 1.5 -> 2)
            scaled = np.floor(scaled / self.unit_step + 0.5 + 1e-9) * self.unit_step
        if self.max_units is not None:
            scaled = np.minimum(scaled, self.max_units)
        return scaled
