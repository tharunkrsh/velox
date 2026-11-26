"""Firm-specific trading-day calendars.

Daily loss limits, trailing drawdowns and consistency rules all depend on which
"trading day" a trade belongs to, and that definition is owned by the prop firm:

* FTMO resets at midnight CE(S)T (``Europe/Prague``, rollover 00:00).
* Topstep's futures session runs 17:00 CT to 16:00 CT the next day, so a trade
  at 18:00 CT on Sunday belongs to Monday (``America/Chicago``, rollover 17:00).

A trading day labelled D therefore covers:
* rollover before noon: [D rollover, D+1 rollover)
* rollover at/after noon (evening session convention): [D-1 rollover, D rollover)

Trades are assigned to the day in which they *close* (when their P&L is realised).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator


class SessionCalendar(BaseModel):
    model_config = ConfigDict(frozen=True)

    timezone: str = "UTC"
    rollover: dt.time = dt.time(0, 0)

    @field_validator("timezone")
    @classmethod
    def _valid_tz(cls, v: str) -> str:
        try:
            pd.Timestamp("2024-01-01", tz=v)
        except Exception as e:  # ZoneInfoNotFoundError is a KeyError, which pydantic won't wrap
            raise ValueError(f"unknown timezone '{v}'") from e
        return v

    def trading_day(self, ts_utc: pd.DatetimeIndex) -> np.ndarray:
        """Map UTC timestamps to trading-day labels (``datetime64[D]``)."""
        local = ts_utc.tz_convert(self.timezone).tz_localize(None)
        offset = pd.Timedelta(hours=self.rollover.hour, minutes=self.rollover.minute,
                              seconds=self.rollover.second)
        days = (local - offset).normalize()
        if self.rollover >= dt.time(12, 0):
            days = days + pd.Timedelta(days=1)
        return days.to_numpy().astype("datetime64[D]")
