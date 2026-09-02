"""Live broker with real exchange connectivity via Binance.

Connects to live Binance spot or futures markets with order execution,
position tracking, and account management.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

from trade.core.secrets import get_broker_config
from trade.core.types import (
    Order,
    OrderSide,
    OrderStatus,
    PortfolioState,
    Position,
)
from trade.execution.broker import Broker

logger = logging.getLogger(__name__)

# Binance order side mappings
SIDE_BUY = "BUY"
SIDE_SELL = "SELL"


class LiveBroker(Broker):
    """Live broker connecting to Binance spot market.

    Handles order execution, position tracking, and portfolio management
    with automatic retry logic and connection error handling.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "",
        testnet: bool = False,
    ) -> None:
        """Initialize Binance connection.

        Args:
            api_key: Binance API key. If empty, loads from TRADE_BROKER_API_KEY env var.
            api_secret: Binance API secret. If empty, loads from TRADE_BROKER_API_SECRET env var.
            base_url: Override base URL (for testnet, use "https://testnet.binance.vision").
            testnet: Whether to use Binance testnet.

        Raises:
            RuntimeError: If credentials are missing or invalid.
        """
        # Load from env if not provided
        if not api_key or not api_secret:
            config = get_broker_config()
            api_key = config.api_key.get_secret_value()
            api_secret = config.api_secret.get_secret_value()
            base_url = config.base_url

        # Validate credentials
        if not api_key or not api_secret:
            raise RuntimeError(
                "LiveBroker requires TRADE_BROKER_API_KEY and TRADE_BROKER_API_SECRET "
                "to be set as environment variables or passed as arguments. "
                "Do not hardcode credentials."
            )

        if len(api_key) < 10 or len(api_secret) < 10:
            raise RuntimeError(
                "API credentials appear invalid. "
                "Check that TRADE_BROKER_API_KEY and TRADE_BROKER_API_SECRET are correct."
            )

        # Initialize Binance client
        self._client = Client(api_key, api_secret, testnet=testnet)
        self._testnet = testnet
        self._base_url = base_url
        self._open_orders: dict[str, Order] = {}
        self._retry_attempts = 3
        self._retry_delay = 0.5  # seconds

        logger.info(
            "LiveBroker connected to Binance %s",
            "testnet" if testnet else "live",
        )

    def submit_order(self, order: Order) -> Order:
        """Submit an order to Binance.

        Args:
            order: Order to submit.

        Returns:
            Order with updated order_id, filled_price, filled_quantity, and status.

        Raises:
            BinanceAPIException: On connection or API errors after retries.
            BinanceOrderException: On order validation errors.
        """
        for attempt in range(self._retry_attempts):
            try:
                side = SIDE_BUY if order.side == OrderSide.LONG else SIDE_SELL
                quantity = self._round_quantity(order.symbol, order.quantity)

                logger.info(
                    "Submitting %s order: %s %.4f @ $%s",
                    side,
                    order.symbol,
                    quantity,
                    order.limit_price or "market",
                )

                # Submit market or limit order
                if order.limit_price:
                    result = self._client.order_limit(
                        symbol=order.symbol,
                        side=side,
                        quantity=quantity,
                        price=order.limit_price,
                    )
                else:
                    result = self._client.order_market(
                        symbol=order.symbol,
                        side=side,
                        quantity=quantity,
                    )

                # Extract fill details
                order.order_id = str(result["orderId"])
                order.filled_quantity = float(result.get("executedQty", 0))

                # Get average fill price from fills array
                fills = result.get("fills", [])
                if fills:
                    total_cost = sum(float(f["qty"]) * float(f["price"]) for f in fills)
                    total_qty = sum(float(f["qty"]) for f in fills)
                    order.filled_price = total_cost / total_qty if total_qty > 0 else 0
                else:
                    order.filled_price = float(result.get("price", 0))

                # Determine status
                if order.filled_quantity == 0:
                    order.status = OrderStatus.REJECTED
                elif order.filled_quantity < quantity * 0.999:  # Allow for float rounding
                    order.status = OrderStatus.PARTIALLY_FILLED
                else:
                    order.status = OrderStatus.FILLED

                logger.info(
                    "Order %s: %s %.4f @ %.8f (status: %s)",
                    order.order_id,
                    order.symbol,
                    order.filled_quantity,
                    order.filled_price,
                    order.status.value,
                )

                self._open_orders[order.order_id] = order
                return order

            except BinanceAPIException as e:
                if attempt < self._retry_attempts - 1:
                    logger.warning(
                        "Binance API error (attempt %d/%d): %s",
                        attempt + 1,
                        self._retry_attempts,
                        e,
                    )
                    time.sleep(self._retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logger.error("Binance order failed after %d retries: %s", self._retry_attempts, e)
                    order.status = OrderStatus.REJECTED
                    return order
            except BinanceOrderException as e:
                logger.error("Binance order validation error: %s", e)
                order.status = OrderStatus.REJECTED
                return order
            except Exception as e:
                logger.error("Unexpected error submitting order: %s", e)
                order.status = OrderStatus.REJECTED
                return order

        order.status = OrderStatus.REJECTED
        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.

        Args:
            order_id: Order ID to cancel.

        Returns:
            True if cancelled successfully, False otherwise.
        """
        if order_id not in self._open_orders:
            logger.warning("Order %s not found in open_orders", order_id)
            return False

        try:
            order = self._open_orders[order_id]
            self._client.cancel_order(symbol=order.symbol, orderId=int(order_id))
            logger.info("Order %s cancelled", order_id)
            del self._open_orders[order_id]
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)
            return False

    def get_positions(self) -> dict[str, Position]:
        """Get current open positions.

        Returns:
            Dictionary mapping symbol to Position objects.
        """
        try:
            account = self._client.get_account()
            positions = {}

            for balance in account.get("balances", []):
                symbol = balance["asset"]
                free = float(balance["free"])
                locked = float(balance["locked"])
                total = free + locked

                if total > 0.00001:  # Ignore dust
                    positions[symbol] = Position(
                        symbol=symbol,
                        quantity=total,
                        entry_price=0.0,  # Not tracked by spot API
                    )

            return positions
        except Exception as e:
            logger.error("Failed to get positions: %s", e)
            return {}

    def get_portfolio(self) -> PortfolioState:
        """Get current portfolio state including cash and positions.

        Returns:
            PortfolioState with current equity, cash, and positions.
        """
        try:
            account = self._client.get_account()
            positions_dict = self.get_positions()

            # Calculate total equity (in USDT)
            total_value = 0.0

            # Add cash (USDT balance)
            for balance in account.get("balances", []):
                if balance["asset"] == "USDT":
                    total_value += float(balance["free"]) + float(balance["locked"])

            # Add value of positions (simplified, assumes USDT pairs)
            # In production, would fetch prices for each position
            # For now, return positions as-is

            return PortfolioState(
                cash=total_value,
                positions=positions_dict,
                total_equity=total_value,
                total_position_value=sum(p.quantity for p in positions_dict.values()),
            )
        except Exception as e:
            logger.error("Failed to get portfolio: %s", e)
            return PortfolioState(cash=0.0, positions={}, total_equity=0.0)

    def get_account_value(self) -> float:
        """Get total account value in USD.

        Returns:
            Total account equity in USDT.
        """
        portfolio = self.get_portfolio()
        return portfolio.total_equity

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to exchange's precision requirements.

        Args:
            symbol: Trading symbol (e.g., BTCUSDT).
            quantity: Desired quantity.

        Returns:
            Quantity rounded to exchange precision.
        """
        try:
            info = self._client.get_symbol_info(symbol)
            for filt in info.get("filters", []):
                if filt["filterType"] == "LOT_SIZE":
                    step_size = float(filt["stepSize"])
                    # Round down to nearest step
                    return float(int(quantity / step_size) * step_size)
        except Exception as e:
            logger.warning("Could not get precision for %s: %s", symbol, e)

        # Default: round to 4 decimals if precision unknown
        return round(quantity, 4)

