"""
[events.py](https://events.py/) — The heartbeat of VELOX.

Every action in the system is represented as an Event.
This decouples data, signals, orders, and fills — exactly
how production trading systems are architected.

Event flow:
MarketEvent → Strategy → SignalEvent
SignalEvent → Portfolio → OrderEvent
OrderEvent  → Execution → FillEvent
FillEvent   → Portfolio → updates positions/P&L
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class EventType(Enum):
	MARKET   = "MARKET"
	SIGNAL   = "SIGNAL"
	ORDER    = "ORDER"
	FILL     = "FILL"


class SignalDirection(Enum):
	LONG  = 1
	SHORT = -1
	EXIT  = 0


class OrderType(Enum):
	MARKET = "MARKET"
	LIMIT  = "LIMIT"


class OrderDirection(Enum):
	BUY  = "BUY"
	SELL = "SELL"


# ─── Events ───────────────────────────────────────────────────────────────────

@dataclass
class MarketEvent:
	"""
	Fired when new OHLCV bar data arrives for one or more symbols.
	Triggers all strategies to re-evaluate signals.
	"""
	type: EventType = field(default=EventType.MARKET, init=False)
	timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SignalEvent:
	"""
	Fired by a Strategy when it detects an alpha opportunity.
	Consumed by the Portfolio to decide whether to act.
	"""
	type: EventType     = field(default=EventType.SIGNAL, init=False)
	timestamp: datetime = field(default_factory=datetime.utcnow)

	symbol: str = ""
	strategy: str = ""  # e.g. "momentum", "pairs"
	direction: SignalDirection = SignalDirection.EXIT
	strength: float = 1.0  # 0.0 → 1.0, used for position sizing
	metadata: dict = field(default_factory=dict)  # strategy-specific extras


@dataclass
class OrderEvent:
	"""
	Fired by the Portfolio when it decides to trade.
	Consumed by the ExecutionHandler.
	"""
	type: EventType = field(default=EventType.ORDER, init=False)
	timestamp: datetime = field(default_factory=datetime.utcnow)

	symbol: str = ""
	order_type: OrderType = OrderType.MARKET
	direction: OrderDirection = OrderDirection.BUY
	quantity: float = 0.0


@dataclass
class FillEvent:
	"""
	Fired by the ExecutionHandler after an order is filled (real or simulated).
	Consumed by the Portfolio to update positions and cash.
	"""
	type: EventType = field(default=EventType.FILL, init=False)
	timestamp: datetime = field(default_factory=datetime.utcnow)

	symbol: str = ""
	direction: OrderDirection = OrderDirection.BUY
	quantity: float = 0.0
	fill_price: float = 0.0  # actual execution price (inc. slippage)
	commission: float = 0.0  # brokerage cost
	exchange: str = "SIM"  # "SIM" for backtests, "ALPACA" for live

	@property
	def fill_cost(self) -> float:
		"""Total cash impact of this fill (excluding commission)."""
		return self.quantity * self.fill_price

	@property
	def total_cost(self) -> float:
		"""Total cash impact including commission."""
		return self.fill_cost + self.commission