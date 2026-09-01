"""Paper trading broker: simulated execution backed by canonical Ledger.

All cash, position, fee, and PnL accounting is delegated to the Ledger.
PaperBroker is a thin adapter between the Broker interface and the
audit-grade Ledger, guaranteeing a single source of truth.
"""

from __future__ import annotations

import datetime as dt
import logging
import uuid

from trade.core.types import Order, OrderSide, OrderStatus, PortfolioState
from trade.core.types import Position as CorePosition
from trade.execution.broker import Broker
from trade.execution.cost_model import CostConfig, CostModel
from trade.execution.ledger import Ledger

logger = logging.getLogger(__name__)


class PaperBroker(Broker):
    """Simulated broker backed by the canonical Ledger.

    Delegates all fill, fee, slippage, cash, and position accounting
    to Ledger so that paper trading results match backtesting,
    training environment, and production paths exactly.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        slippage_pct: float = 0.0005,
        commission_pct: float = 0.001,
        cost_config: CostConfig | None = None,
    ) -> None:
        cc = cost_config or CostConfig(
            taker_fee=commission_pct,
            maker_fee=commission_pct,
            entry_slippage=slippage_pct,
            exit_slippage=slippage_pct,
        )
        self._ledger = Ledger(initial_capital, cc)
        self._order_history: list[Order] = []
        self._current_prices: dict[str, float] = {}

    # -- Price feed -----------------------------------------------------------

    def set_price(self, symbol: str, price: float) -> None:
        """Update the current market price for a symbol."""
        self._current_prices[symbol] = price

    # -- Broker interface -----------------------------------------------------

    def submit_order(self, order: Order) -> Order:
        """Simulate order execution via canonical Ledger."""
        order.order_id = str(uuid.uuid4())[:8]

        price = self._current_prices.get(order.symbol, 0.0)
        if price <= 0:
            order.status = OrderStatus.REJECTED
            logger.warning("Paper order rejected: no price for %s", order.symbol)
            self._order_history.append(order)
            return order

        side = "BUY" if order.side == OrderSide.LONG else "SELL"

        # Determine intent: open new position, or close existing
        open_pos = self._ledger.open_position
        has_position = open_pos is not None

        if order.side == OrderSide.LONG and not has_position:
            # Open long
            success = self._ledger.enter_position(
                order.symbol, "BUY", order.quantity, price
            )
            if not success:
                order.status = OrderStatus.REJECTED
                logger.warning("Paper order rejected by Ledger (insufficient cash or position exists)")
                self._order_history.append(order)
                return order
            # Get actual fill price from position
            new_pos = self._ledger.open_position
            order.filled_price = new_pos.entry_price if new_pos else price

        elif order.side == OrderSide.SHORT and has_position and open_pos.side == "BUY":
            # Close long
            trade = self._ledger.close_position(price)
            if trade is None:
                order.status = OrderStatus.REJECTED
                self._order_history.append(order)
                return order
            order.filled_price = trade.exit_price

        elif order.side == OrderSide.SHORT and not has_position:
            # Open short
            success = self._ledger.enter_position(
                order.symbol, "SELL", order.quantity, price
            )
            if not success:
                order.status = OrderStatus.REJECTED
                logger.warning("Paper order rejected by Ledger (position exists)")
                self._order_history.append(order)
                return order
            new_pos = self._ledger.open_position
            order.filled_price = new_pos.entry_price if new_pos else price

        elif order.side == OrderSide.LONG and has_position and open_pos.side == "SELL":
            # Close short (buy to cover)
            trade = self._ledger.close_position(price)
            if trade is None:
                order.status = OrderStatus.REJECTED
                self._order_history.append(order)
                return order
            order.filled_price = trade.exit_price

        else:
            # Adding to an existing position in the same direction is not
            # supported by the single-position Ledger; reject.
            order.status = OrderStatus.REJECTED
            logger.warning("Paper order rejected: cannot add to existing position")
            self._order_history.append(order)
            return order

        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.timestamp = dt.datetime.utcnow()
        self._order_history.append(order)

        logger.info(
            "Paper %s: %s %.2f @ %.4f",
            order.side.value, order.symbol, order.quantity, order.filled_price,
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Paper orders are instant — nothing to cancel."""
        return False

    def get_positions(self) -> dict[str, CorePosition]:
        pos = self._ledger.open_position
        if pos is None:
            return {}
        return {
            pos.symbol: CorePosition(
                symbol=pos.symbol,
                quantity=pos.quantity,
                avg_entry_price=pos.entry_price,
                current_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl(),
                entry_fees=pos.entry_fee,
            ),
        }

    def get_portfolio(self) -> PortfolioState:
        snap = self._ledger.snapshot(self._current_prices.get(
            self._ledger.open_position.symbol if self._ledger.open_position else "", 0.0
        ))
        return PortfolioState(
            cash=snap.cash,
            positions=self.get_positions(),
            total_equity=snap.equity,
        )

    def get_account_value(self) -> float:
        return self.get_portfolio().total_equity

    @property
    def ledger(self) -> Ledger:
        """Expose the canonical ledger for direct access."""
        return self._ledger

    @property
    def order_history(self) -> list[Order]:
        return self._order_history.copy()
