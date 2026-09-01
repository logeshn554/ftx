"""Risk engine: the gatekeeper between AI decisions and order execution.

Every order proposed by the AI agent passes through this engine.
The risk engine can MODIFY (reduce size) or REJECT orders entirely.
The risk engine has HIGHER AUTHORITY than the AI.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import deque
from typing import Any

from trade.core.events import RiskBreached, event_bus
from trade.core.types import (
    Order,
    OrderSide,
    OrderStatus,
    PortfolioState,
    RiskDecision,
)
from trade.risk.limits import RiskLimits

logger = logging.getLogger(__name__)


class RiskEngine:
    """Evaluates proposed orders against configured risk limits.

    The engine runs a chain of checks in order. Any single check failure
    can modify or reject the order. All checks are logged for audit.

    Check chain:
        1. Trading enabled?
        2. Data freshness valid?
        3. Daily loss limit exceeded?
        4. Position size allowed?
        5. Leverage allowed?
        6. Order rate limit?
        7. Order value cap?
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self.limits = limits or RiskLimits()
        self._trading_enabled = True
        self._daily_start_equity: float = 0.0
        self._daily_pnl: float = 0.0
        self._order_timestamps: deque[dt.datetime] = deque(maxlen=1000)
        self._daily_order_count: int = 0
        self._consecutive_losses: int = 0
        self._last_data_timestamp: dt.datetime | None = None
        self._audit_log: list[dict[str, Any]] = []

    def evaluate(
        self,
        order: Order,
        portfolio: PortfolioState,
        current_price: float,
        data_timestamp: dt.datetime | None = None,
    ) -> RiskDecision:
        """Evaluate a proposed order against all risk limits.

        Args:
            order: The order proposed by the AI agent.
            portfolio: Current portfolio state.
            current_price: Latest market price for the symbol.
            data_timestamp: Timestamp of the market data used for this decision.

        Returns:
            RiskDecision indicating whether the order is approved, modified, or rejected.
        """
        rejections: list[str] = []
        warnings: list[str] = []
        modified_order = Order(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.limit_price,
        )

        # --- Check 1: Trading enabled? ---
        if not self._trading_enabled:
            rejections.append("TRADING_DISABLED: Trading is currently disabled")

        # --- Check 2: Data freshness ---
        if data_timestamp is not None:
            age = (dt.datetime.utcnow() - data_timestamp).total_seconds()
            if age > self.limits.min_data_freshness_seconds:
                rejections.append(
                    f"STALE_DATA: Market data is {age:.0f}s old "
                    f"(limit: {self.limits.min_data_freshness_seconds:.0f}s)"
                )

        # --- Check 3: Daily loss limit ---
        if self._daily_start_equity > 0:
            daily_loss_pct = abs(min(0, self._daily_pnl)) / self._daily_start_equity * 100
            if daily_loss_pct >= self.limits.max_daily_loss_pct:
                rejections.append(
                    f"DAILY_LOSS_LIMIT: Daily loss {daily_loss_pct:.2f}% "
                    f"exceeds limit {self.limits.max_daily_loss_pct:.2f}%"
                )

        # --- Check 4: Position size ---
        order_value = modified_order.quantity * current_price
        max_position_value = portfolio.total_equity * (self.limits.max_position_pct / 100)

        if order_value > max_position_value:
            # MODIFY: reduce position size
            old_qty = modified_order.quantity
            modified_order.quantity = max_position_value / current_price
            warnings.append(
                f"POSITION_SIZE_REDUCED: {old_qty:.2f} → {modified_order.quantity:.2f} shares "
                f"(max {self.limits.max_position_pct}% of equity)"
            )

        # Check number of open positions
        if len(portfolio.positions) >= self.limits.max_open_positions:
            if order.side == OrderSide.LONG and order.symbol not in portfolio.positions:
                rejections.append(
                    f"MAX_POSITIONS: {len(portfolio.positions)} open positions "
                    f"(limit: {self.limits.max_open_positions})"
                )

        # --- Check 5: Leverage ---
        total_exposure = portfolio.total_position_value + order_value
        if portfolio.total_equity > 0:
            leverage = total_exposure / portfolio.total_equity
            if leverage > self.limits.max_leverage:
                rejections.append(
                    f"LEVERAGE_EXCEEDED: Effective leverage {leverage:.2f}x "
                    f"exceeds limit {self.limits.max_leverage:.2f}x"
                )

        # --- Check 6: Order rate limit ---
        now = dt.datetime.utcnow()
        recent_orders = sum(
            1 for ts in self._order_timestamps
            if (now - ts).total_seconds() < 60
        )
        if recent_orders >= self.limits.max_orders_per_minute:
            rejections.append(
                f"RATE_LIMIT: {recent_orders} orders in last minute "
                f"(limit: {self.limits.max_orders_per_minute})"
            )

        if self._daily_order_count >= self.limits.max_orders_per_day:
            rejections.append(
                f"DAILY_ORDER_LIMIT: {self._daily_order_count} orders today "
                f"(limit: {self.limits.max_orders_per_day})"
            )

        # --- Check 7: Order value cap ---
        adjusted_value = modified_order.quantity * current_price
        if adjusted_value > self.limits.max_order_value:
            old_qty = modified_order.quantity
            modified_order.quantity = self.limits.max_order_value / current_price
            warnings.append(
                f"ORDER_VALUE_CAPPED: {old_qty:.2f} → {modified_order.quantity:.2f} shares "
                f"(max order value ${self.limits.max_order_value:,.0f})"
            )

        # --- Build decision ---
        approved = len(rejections) == 0
        was_modified = modified_order.quantity != order.quantity

        decision = RiskDecision(
            approved=approved,
            original_order=order,
            modified_order=modified_order if was_modified else None,
            rejections=rejections,
            warnings=warnings,
        )

        # Track order if approved
        if approved:
            self._order_timestamps.append(now)
            self._daily_order_count += 1

        # Audit log
        self._log_decision(decision, current_price)

        # Emit events for rejections
        if not approved:
            for rejection in rejections:
                event_bus.publish_sync(
                    RiskBreached(
                        limit_name=rejection.split(":")[0],
                        action_taken="REJECTED",
                    )
                )

        return decision

    def update_daily_pnl(self, pnl: float) -> None:
        """Update the running daily PnL tracker."""
        self._daily_pnl = pnl

    def reset_daily(self, starting_equity: float) -> None:
        """Reset daily tracking at market open."""
        self._daily_start_equity = starting_equity
        self._daily_pnl = 0.0
        self._daily_order_count = 0
        logger.info("Daily risk counters reset. Starting equity: $%.2f", starting_equity)

    def record_trade_result(self, pnl: float) -> None:
        """Record a trade result for consecutive loss tracking."""
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        if self._consecutive_losses >= self.limits.consecutive_loss_limit:
            logger.warning(
                "%d consecutive losses — consider circuit breaker",
                self._consecutive_losses,
            )

    def enable_trading(self) -> None:
        """Enable trading."""
        self._trading_enabled = True
        logger.info("Trading ENABLED")

    def disable_trading(self) -> None:
        """Disable trading — all orders will be rejected."""
        self._trading_enabled = False
        logger.warning("Trading DISABLED")

    @property
    def is_trading_enabled(self) -> bool:
        return self._trading_enabled

    @property
    def audit_log(self) -> list[dict[str, Any]]:
        return self._audit_log.copy()

    def _log_decision(self, decision: RiskDecision, price: float) -> None:
        """Log a risk decision to the audit trail."""
        entry = {
            "timestamp": dt.datetime.utcnow().isoformat(),
            "symbol": decision.original_order.symbol,
            "side": decision.original_order.side.value,
            "requested_qty": decision.original_order.quantity,
            "approved": decision.approved,
            "price": price,
        }

        if decision.modified_order:
            entry["modified_qty"] = decision.modified_order.quantity

        if decision.rejections:
            entry["rejections"] = decision.rejections

        if decision.warnings:
            entry["warnings"] = decision.warnings

        self._audit_log.append(entry)

        if not decision.approved:
            logger.warning(
                "ORDER REJECTED: %s %s %.2f | %s",
                decision.original_order.side.value,
                decision.original_order.symbol,
                decision.original_order.quantity,
                "; ".join(decision.rejections),
            )
        elif decision.warnings:
            logger.info(
                "ORDER MODIFIED: %s %s | %s",
                decision.original_order.side.value,
                decision.original_order.symbol,
                "; ".join(decision.warnings),
            )
