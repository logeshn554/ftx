"""Configurable risk limit definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskLimits:
    """Defines all risk thresholds enforced by the risk engine.

    The risk engine has **higher authority** than the AI agent.
    Any order that violates these limits will be modified or rejected.
    """

    # Position sizing
    max_position_pct: float = 20.0
    """Maximum % of total equity in a single position."""

    max_open_positions: int = 10
    """Maximum number of concurrent open positions."""

    # Loss limits
    max_daily_loss_pct: float = 5.0
    """Maximum allowed daily portfolio loss as % of starting equity."""

    max_weekly_loss_pct: float = 10.0
    """Maximum allowed weekly portfolio loss."""

    max_total_drawdown_pct: float = 25.0
    """Maximum total drawdown before circuit breaker trips."""

    # Order limits
    max_order_value: float = 50_000.0
    """Absolute cap on single order value in account currency."""

    max_order_pct: float = 10.0
    """Maximum single order as % of total equity."""

    # Leverage
    max_leverage: float = 1.0
    """Maximum leverage ratio (1.0 = no leverage)."""

    # Data quality
    min_data_freshness_seconds: float = 300.0
    """Reject orders if market data is older than this (seconds)."""

    # Rate limiting
    max_orders_per_minute: int = 10
    """Maximum orders per minute to prevent runaway algorithms."""

    max_orders_per_day: int = 100
    """Maximum total orders per day."""

    # Circuit breaker
    circuit_breaker_cooldown_seconds: float = 3600.0
    """Cooldown period after circuit breaker trips (seconds)."""

    consecutive_loss_limit: int = 5
    """Trip circuit breaker after N consecutive losing trades."""

    def validate(self) -> list[str]:
        """Validate that limits are internally consistent.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        if self.max_position_pct <= 0 or self.max_position_pct > 100:
            errors.append(f"max_position_pct must be in (0, 100], got {self.max_position_pct}")
        if self.max_daily_loss_pct <= 0:
            errors.append(f"max_daily_loss_pct must be > 0, got {self.max_daily_loss_pct}")
        if self.max_leverage < 1.0:
            errors.append(f"max_leverage must be >= 1.0, got {self.max_leverage}")
        if self.max_order_value <= 0:
            errors.append(f"max_order_value must be > 0, got {self.max_order_value}")
        if self.max_orders_per_minute <= 0:
            errors.append(f"max_orders_per_minute must be > 0, got {self.max_orders_per_minute}")

        return errors
