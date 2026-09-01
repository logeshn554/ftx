"""Shadow broker: logs orders but does NOT execute them.

Used to validate a candidate model's decisions in production conditions
without any financial risk.
"""

from __future__ import annotations

import datetime as dt
import logging

from trade.core.types import Order, OrderStatus, PortfolioState, Position
from trade.execution.broker import Broker

logger = logging.getLogger(__name__)


class ShadowBroker(Broker):
    """Shadow mode broker — records orders without execution.

    Computes hypothetical PnL and positions based on what would have
    happened if the orders were executed, for comparison against
    the production model.
    """

    def __init__(self) -> None:
        self._shadow_log: list[dict] = []

    def submit_order(self, order: Order) -> Order:
        """Log the order but do NOT execute."""
        order.status = OrderStatus.CANCELLED  # Not actually executed

        self._shadow_log.append({
            "timestamp": dt.datetime.utcnow().isoformat(),
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "order_type": order.order_type.value,
            "status": "SHADOW_LOGGED",
        })

        logger.info(
            "SHADOW: %s %s %.2f (logged only, not executed)",
            order.side.value, order.symbol, order.quantity,
        )
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_positions(self) -> dict[str, Position]:
        return {}

    def get_portfolio(self) -> PortfolioState:
        return PortfolioState(cash=0.0, total_equity=0.0)

    def get_account_value(self) -> float:
        return 0.0

    @property
    def shadow_log(self) -> list[dict]:
        return self._shadow_log.copy()
