"""Live broker stub for real exchange connectivity.

This module defines the interface for connecting to a live broker
(Alpaca, Interactive Brokers, etc.). The actual implementation will
be added when progressing to Stage 7-8 (Shadow/Live Trading).
"""

from __future__ import annotations

import logging

from trade.core.types import Order, OrderStatus, PortfolioState, Position
from trade.execution.broker import Broker

logger = logging.getLogger(__name__)


class LiveBroker(Broker):
    """Live broker — connects to a real exchange.

    NOT YET IMPLEMENTED. This stub ensures type-safety and allows
    the rest of the system to be developed against the Broker interface.

    Target integrations:
        - Alpaca (US equities)
        - Interactive Brokers (multi-asset)
        - Binance (crypto)
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "",
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url
        logger.warning("LiveBroker initialized but NOT IMPLEMENTED — do not use for real trading")

    def submit_order(self, order: Order) -> Order:
        raise NotImplementedError(
            "LiveBroker is not yet implemented. "
            "Use PaperBroker or ShadowBroker for testing."
        )

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("LiveBroker not yet implemented")

    def get_positions(self) -> dict[str, Position]:
        raise NotImplementedError("LiveBroker not yet implemented")

    def get_portfolio(self) -> PortfolioState:
        raise NotImplementedError("LiveBroker not yet implemented")

    def get_account_value(self) -> float:
        raise NotImplementedError("LiveBroker not yet implemented")
