"""Shadow broker: logs orders but does NOT execute them.

Used to validate a candidate model's decisions in production conditions
without any financial risk. Tracks hypothetical PnL for comparison.
"""

from __future__ import annotations

import datetime as dt
import logging

from trade.core.types import Order, OrderStatus, PortfolioState, Position, OrderSide
from trade.execution.broker import Broker

logger = logging.getLogger(__name__)


class ShadowBroker(Broker):
    """Shadow mode broker — records orders and tracks hypothetical PnL.

    Simulates fills at logged prices and computes hypothetical positions
    and PnL based on what would have happened if orders were executed,
    for comparison against the production model.
    """

    def __init__(self, initial_capital: float = 100_000.0) -> None:
        self._shadow_log: list[dict] = []
        self._simulated_positions: dict[str, dict] = {}  # symbol -> {qty, entry_price}
        self._simulated_cash = initial_capital
        self._initial_capital = initial_capital
        self._hypothetical_pnl = 0.0
        self._peak_equity = initial_capital
        logger.info("ShadowBroker initialized with $%.2f", initial_capital)

    def submit_order(self, order: Order) -> Order:
        """Log the order and simulate a fill."""
        order.status = OrderStatus.CANCELLED  # Mark as shadow (not real)

        # Simulate fill at provided limit price or 0 (market)
        fill_price = order.limit_price or 0.0
        if fill_price <= 0:
            # For market orders, we need to know current price
            # In real usage, pass this in or fetch from external source
            fill_price = 100.0  # Default placeholder

        order.filled_price = fill_price
        order.filled_quantity = order.quantity

        # Update simulated cash and positions
        cost = order.quantity * fill_price

        if order.side == OrderSide.LONG:
            self._simulated_cash -= cost
            if order.symbol in self._simulated_positions:
                pos = self._simulated_positions[order.symbol]
                # Average entry price
                total_qty = pos["qty"] + order.quantity
                total_cost = (pos["qty"] * pos["entry_price"]) + cost
                pos["qty"] = total_qty
                pos["entry_price"] = total_cost / total_qty if total_qty > 0 else 0
            else:
                self._simulated_positions[order.symbol] = {
                    "qty": order.quantity,
                    "entry_price": fill_price,
                }
        elif order.side == OrderSide.SHORT:
            # For short sales, we add to cash (simplified)
            self._simulated_cash += cost
            if order.symbol in self._simulated_positions:
                pos = self._simulated_positions[order.symbol]
                pos["qty"] -= order.quantity
                if pos["qty"] <= 0:
                    del self._simulated_positions[order.symbol]
            else:
                self._simulated_positions[order.symbol] = {
                    "qty": -order.quantity,
                    "entry_price": fill_price,
                }

        # Log entry
        self._shadow_log.append({
            "timestamp": dt.datetime.utcnow().isoformat(),
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "fill_price": fill_price,
            "order_type": order.order_type.value,
            "status": "SHADOW_LOGGED",
            "simulated_cash_after": self._simulated_cash,
        })

        logger.info(
            "SHADOW: %s %s %.2f @ $%.2f (cash now: $%.2f)",
            order.side.value,
            order.symbol,
            order.quantity,
            fill_price,
            self._simulated_cash,
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Shadow broker cannot cancel (no real orders)."""
        return True

    def get_positions(self) -> dict[str, Position]:
        """Get hypothetical simulated positions."""
        positions = {}
        for symbol, pos_data in self._simulated_positions.items():
            if pos_data["qty"] != 0:
                positions[symbol] = Position(
                    symbol=symbol,
                    quantity=pos_data["qty"],
                    entry_price=pos_data["entry_price"],
                )
        return positions

    def get_portfolio(self) -> PortfolioState:
        """Get hypothetical simulated portfolio state."""
        positions = self.get_positions()

        # Calculate position value (simplified, assumes last entry price)
        total_position_value = sum(
            p.quantity * p.entry_price
            for p in positions.values()
        )

        total_equity = self._simulated_cash + total_position_value

        return PortfolioState(
            cash=self._simulated_cash,
            positions=positions,
            total_equity=total_equity,
            total_position_value=total_position_value,
        )

    def get_account_value(self) -> float:
        """Get total hypothetical account value."""
        portfolio = self.get_portfolio()
        return portfolio.total_equity

    def set_position_price(self, symbol: str, current_price: float) -> None:
        """Update current market price for a position (for PnL calculation).

        Args:
            symbol: Trading symbol.
            current_price: Current market price.
        """
        if symbol in self._simulated_positions:
            pos = self._simulated_positions[symbol]
            pnl = pos["qty"] * (current_price - pos["entry_price"])
            self._hypothetical_pnl += pnl
            logger.debug(
                "SHADOW PnL update: %s %.2f @ $%.2f → $%.2f (PnL: $%.2f)",
                symbol,
                pos["qty"],
                pos["entry_price"],
                current_price,
                pnl,
            )

    def update_peak_equity(self) -> None:
        """Track peak equity for drawdown calculation."""
        current_equity = self.get_account_value()
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    @property
    def shadow_log(self) -> list[dict]:
        """Get shadow order log."""
        return self._shadow_log.copy()

    @property
    def hypothetical_pnl(self) -> float:
        """Get cumulative hypothetical PnL."""
        return self._hypothetical_pnl

    @property
    def hypothetical_return(self) -> float:
        """Get hypothetical return as percentage."""
        if self._initial_capital <= 0:
            return 0.0
        return (self.get_account_value() - self._initial_capital) / self._initial_capital * 100
