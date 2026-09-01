"""Abstract broker interface.

All execution modes (paper, shadow, live) implement this interface
so the trading loop is agnostic to the execution backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from trade.core.types import Order, PortfolioState, Position


class Broker(ABC):
    """Abstract broker interface for order execution."""

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order for execution.

        Returns:
            The order with updated status and fill information.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if cancelled."""

    @abstractmethod
    def get_positions(self) -> dict[str, Position]:
        """Get all current positions."""

    @abstractmethod
    def get_portfolio(self) -> PortfolioState:
        """Get current portfolio state."""

    @abstractmethod
    def get_account_value(self) -> float:
        """Get total account value."""
