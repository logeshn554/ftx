"""Paper trading broker: simulated execution with configurable slippage."""

from __future__ import annotations

import datetime as dt
import logging
import uuid

from trade.core.types import Order, OrderSide, OrderStatus, PortfolioState, Position
from trade.execution.broker import Broker

logger = logging.getLogger(__name__)


class PaperBroker(Broker):
    """Simulated broker for paper trading.

    Executes orders instantly with configurable slippage and latency.
    Maintains a virtual portfolio with cash and positions.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.0005,
        commission_pct: float = 0.001,
    ) -> None:
        self._cash = initial_capital
        self._initial_capital = initial_capital
        self._positions: dict[str, Position] = {}
        self._order_history: list[Order] = []
        self._slippage_pct = slippage_pct
        self._commission_pct = commission_pct
        self._current_prices: dict[str, float] = {}

    def set_price(self, symbol: str, price: float) -> None:
        """Update the current market price for a symbol."""
        self._current_prices[symbol] = price
        # Update position prices
        if symbol in self._positions:
            pos = self._positions[symbol]
            pos.current_price = price
            # Mark to the observable market price; do not use a future fill.
            pos.unrealized_pnl = (price - pos.avg_entry_price) * pos.quantity - pos.entry_fees

    def submit_order(self, order: Order) -> Order:
        """Simulate order execution."""
        order.order_id = str(uuid.uuid4())[:8]

        price = self._current_prices.get(order.symbol, 0.0)
        if price <= 0:
            order.status = OrderStatus.REJECTED
            logger.warning("Paper order rejected: no price for %s", order.symbol)
            self._order_history.append(order)
            return order

        # Apply slippage
        if order.side == OrderSide.LONG:
            fill_price = price * (1 + self._slippage_pct)
        else:
            fill_price = price * (1 - self._slippage_pct)

        order_value = order.quantity * fill_price
        commission = order_value * self._commission_pct

        if order.side == OrderSide.LONG:
            # Check cash
            if self._cash < order_value + commission:
                order.status = OrderStatus.REJECTED
                logger.warning("Paper order rejected: insufficient cash")
                self._order_history.append(order)
                return order

            self._cash -= (order_value + commission)

            # Update or create position
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                total_cost = pos.avg_entry_price * pos.quantity + fill_price * order.quantity
                total_qty = pos.quantity + order.quantity
                pos.avg_entry_price = total_cost / total_qty
                pos.quantity = total_qty
                pos.entry_fees += commission
                pos.slippage_cost += abs(fill_price - price) * order.quantity
            else:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_entry_price=fill_price,
                    current_price=price,
                    entry_fees=commission,
                    slippage_cost=abs(fill_price - price) * order.quantity,
                )

        elif order.side == OrderSide.SHORT:
            # Selling — close position
            if order.symbol not in self._positions:
                order.status = OrderStatus.REJECTED
                logger.warning("Paper order rejected: no position to sell")
                self._order_history.append(order)
                return order

            pos = self._positions[order.symbol]
            sell_qty = min(order.quantity, pos.quantity)
            # Allocate entry fees for partial closes, then recognize net PnL
            # from actual entry/exit fills only.
            entry_fee = pos.entry_fees * sell_qty / pos.quantity
            proceeds = sell_qty * fill_price - commission

            pos.realized_pnl += (fill_price - pos.avg_entry_price) * sell_qty - entry_fee - commission
            pos.entry_fees -= entry_fee
            pos.quantity -= sell_qty
            self._cash += proceeds

            if pos.quantity <= 0:
                del self._positions[order.symbol]

        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_quantity = order.quantity
        order.timestamp = dt.datetime.utcnow()
        self._order_history.append(order)

        logger.info(
            "Paper %s: %s %.2f @ %.2f (commission: %.2f)",
            order.side.value, order.symbol, order.quantity, fill_price, commission,
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Paper orders are instant — nothing to cancel."""
        return False

    def get_positions(self) -> dict[str, Position]:
        return self._positions.copy()

    def get_portfolio(self) -> PortfolioState:
        total_position_value = sum(
            p.quantity * p.current_price for p in self._positions.values()
        )
        total_equity = self._cash + total_position_value

        return PortfolioState(
            cash=self._cash,
            positions=self._positions.copy(),
            total_equity=total_equity,
        )

    def get_account_value(self) -> float:
        return self._cash + sum(
            p.quantity * p.current_price for p in self._positions.values()
        )

    @property
    def order_history(self) -> list[Order]:
        return self._order_history.copy()
