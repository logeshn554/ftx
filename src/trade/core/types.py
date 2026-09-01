"""Shared types, enums, and dataclasses used across all components."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


# ---------------------------------------------------------------------------
# Trading Actions
# ---------------------------------------------------------------------------

class Action(IntEnum):
    """Discrete action space for the RL agent."""

    HOLD = 0
    BUY = 1
    SELL = 2


class OrderSide(str, Enum):
    """Order direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(str, Enum):
    """Order execution type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    """Lifecycle status of an order."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# Market Regime
# ---------------------------------------------------------------------------

class Regime(str, Enum):
    """Detected market regime."""

    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


# ---------------------------------------------------------------------------
# Model Lifecycle
# ---------------------------------------------------------------------------

class ModelStage(str, Enum):
    """Stage in the model promotion pipeline."""

    CANDIDATE = "candidate"
    BACKTESTING = "backtesting"
    PAPER_TRADING = "paper_trading"
    SHADOW = "shadow"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    REJECTED = "rejected"


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"          # Normal operation
    OPEN = "OPEN"              # Trading halted
    HALF_OPEN = "HALF_OPEN"    # Testing with reduced limits


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelVersion:
    """Immutable descriptor for a model checkpoint."""

    major: int
    minor: int
    patch: int
    created_at: dt.datetime = field(default_factory=dt.datetime.utcnow)
    stage: ModelStage = ModelStage.CANDIDATE
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    dataset_hash: str = ""
    notes: str = ""

    @property
    def tag(self) -> str:
        return f"v{self.major}.{self.minor}.{self.patch}"

    def __str__(self) -> str:
        return self.tag


@dataclass
class Order:
    """Represents a trading order proposed by the agent or modified by risk."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float | None = None
    filled_quantity: float = 0.0
    timestamp: dt.datetime = field(default_factory=dt.datetime.utcnow)
    order_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current position in a single instrument."""

    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_fees: float = 0.0
    slippage_cost: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_entry_price


@dataclass
class PortfolioState:
    """Snapshot of full portfolio state."""

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    total_equity: float = 0.0
    daily_pnl: float = 0.0
    daily_return: float = 0.0
    timestamp: dt.datetime = field(default_factory=dt.datetime.utcnow)

    @property
    def total_position_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())


@dataclass
class RiskDecision:
    """Result of the risk engine evaluating a proposed order."""

    approved: bool
    original_order: Order
    modified_order: Order | None = None
    rejections: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp: dt.datetime = field(default_factory=dt.datetime.utcnow)


@dataclass
class TrainingResult:
    """Output from a training run."""

    model_version: ModelVersion
    total_timesteps: int
    training_time_seconds: float
    final_metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str = ""
    mlflow_run_id: str = ""


@dataclass
class BacktestResult:
    """Output from a backtest run."""

    model_version: ModelVersion
    start_date: dt.date
    end_date: dt.date
    initial_capital: float
    final_capital: float
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    transaction_costs: float = 0.0
    daily_returns: list[float] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Result of comparing two models."""

    class Verdict(str, Enum):
        PROMOTE = "PROMOTE"
        REJECT = "REJECT"
        INCONCLUSIVE = "INCONCLUSIVE"

    champion: ModelVersion
    challenger: ModelVersion
    verdict: Verdict
    champion_metrics: dict[str, float] = field(default_factory=dict)
    challenger_metrics: dict[str, float] = field(default_factory=dict)
    improvements: dict[str, float] = field(default_factory=dict)
    regressions: dict[str, float] = field(default_factory=dict)
    statistical_significance: dict[str, float] = field(default_factory=dict)
    notes: str = ""
