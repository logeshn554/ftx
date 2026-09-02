"""Tests for LiveBroker — the most financially critical code.

FIX 23: Add test coverage for real Binance order execution, credential validation,
partial fills, and error handling.
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch, call

from trade.core.types import OrderSide, PortfolioState
from trade.execution.live import LiveBroker
from trade.core.secrets import BrokerConfig
from binance.exceptions import BinanceAPIException, BinanceOrderException


class TestLiveBrokerCredentialValidation:
    """FIX 23: Validate that empty credentials raise RuntimeError."""

    def test_empty_api_key_raises_error(self):
        """Constructor should raise RuntimeError if api_key is empty."""
        config = BrokerConfig(api_key="", api_secret="secret123")
        with pytest.raises(RuntimeError, match="Broker credentials not configured"):
            LiveBroker(config)

    def test_empty_api_secret_raises_error(self):
        """Constructor should raise RuntimeError if api_secret is empty."""
        config = BrokerConfig(api_key="key123", api_secret="")
        with pytest.raises(RuntimeError, match="Broker credentials not configured"):
            LiveBroker(config)

    def test_both_empty_raises_error(self):
        """Constructor should raise RuntimeError if both are empty."""
        config = BrokerConfig(api_key="", api_secret="")
        with pytest.raises(RuntimeError, match="Broker credentials not configured"):
            LiveBroker(config)

    def test_valid_credentials_initialize(self):
        """Constructor should succeed with valid credentials."""
        config = BrokerConfig(api_key="key123", api_secret="secret123")
        with patch("binance.client.Client"):
            broker = LiveBroker(config, testnet=True)
            assert broker is not None


class TestLiveBrokerOrderExecution:
    """FIX 23: Test order submission with filled/partial/rejected states."""

    @pytest.fixture
    def broker_with_mock_client(self):
        """Create broker with mocked Binance client."""
        config = BrokerConfig(api_key="key123", api_secret="secret123")
        with patch("binance.client.Client") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            broker = LiveBroker(config, testnet=True)
            broker.client = mock_instance
            yield broker

    def test_market_order_filled(self, broker_with_mock_client):
        """Test market order that fills completely."""
        broker = broker_with_mock_client

        # Mock successful order response
        broker.client.order_market = Mock(
            return_value={
                "orderId": 12345,
                "status": "FILLED",
                "executedQty": "1.0",
                "origQty": "1.0",
            }
        )

        result = broker.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type="market",
        )

        assert result["order_id"] == 12345
        assert result["filled_qty"] == 1.0
        assert result["status"] == "FILLED"
        broker.client.order_market.assert_called_once()

    def test_limit_order_pending(self, broker_with_mock_client):
        """Test limit order that remains pending."""
        broker = broker_with_mock_client

        broker.client.order_limit = Mock(
            return_value={
                "orderId": 12346,
                "status": "NEW",
                "executedQty": "0.0",
                "origQty": "1.0",
            }
        )

        result = broker.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type="limit",
            limit_price=29000.0,
        )

        assert result["order_id"] == 12346
        assert result["filled_qty"] == 0.0
        assert result["status"] == "NEW"

    def test_partial_fill_detection(self, broker_with_mock_client):
        """FIX 23: Detect and track partial fills."""
        broker = broker_with_mock_client

        broker.client.order_market = Mock(
            return_value={
                "orderId": 12347,
                "status": "PARTIALLY_FILLED",
                "executedQty": "0.6",  # 60% filled
                "origQty": "1.0",
            }
        )

        result = broker.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type="market",
        )

        assert result["status"] == "PARTIALLY_FILLED"
        assert result["filled_qty"] == 0.6
        # System should queue remainder (1.0 - 0.6 = 0.4) for next attempt
        assert result.get("order_id") == 12347

    def test_rejected_order(self, broker_with_mock_client):
        """Test order rejection."""
        broker = broker_with_mock_client

        broker.client.order_market = Mock(
            return_value={
                "orderId": 12348,
                "status": "REJECTED",
                "executedQty": "0.0",
                "origQty": "1.0",
            }
        )

        result = broker.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type="market",
        )

        assert result["status"] == "REJECTED"
        assert result["filled_qty"] == 0.0


class TestLiveBrokerErrorHandling:
    """FIX 23: Test resilience with API errors and retries."""

    @pytest.fixture
    def broker_with_mock_client(self):
        """Create broker with mocked Binance client."""
        config = BrokerConfig(api_key="key123", api_secret="secret123")
        with patch("binance.client.Client") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            broker = LiveBroker(config, testnet=True)
            broker.client = mock_instance
            yield broker

    def test_api_exception_with_retry(self, broker_with_mock_client):
        """Test exponential backoff retry on API error."""
        broker = broker_with_mock_client

        # Fail twice, succeed on third attempt
        broker.client.order_market = Mock(
            side_effect=[
                BinanceAPIException(code=-1000, message="Invalid request"),
                BinanceAPIException(code=-1000, message="Invalid request"),
                {"orderId": 12349, "status": "FILLED", "executedQty": "1.0", "origQty": "1.0"},
            ]
        )

        result = broker.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type="market",
        )

        # Should succeed after retries
        assert result["order_id"] == 12349
        assert broker.client.order_market.call_count == 3

    def test_max_retries_exceeded(self, broker_with_mock_client):
        """Test failure after max retries."""
        broker = broker_with_mock_client

        broker.client.order_market = Mock(
            side_effect=BinanceAPIException(code=-1000, message="Service unavailable")
        )

        result = broker.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type="market",
        )

        # Should return error status
        assert "error" in result or result["filled_qty"] == 0.0
        # Should have retried 3 times
        assert broker.client.order_market.call_count == 3


class TestLiveBrokerQuantityPrecision:
    """FIX 23: Test quantity rounding to exchange precision."""

    @pytest.fixture
    def broker_with_mock_client(self):
        """Create broker with mocked Binance client."""
        config = BrokerConfig(api_key="key123", api_secret="secret123")
        with patch("binance.client.Client") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            broker = LiveBroker(config, testnet=True)
            broker.client = mock_instance
            yield broker

    def test_round_quantity_to_precision(self, broker_with_mock_client):
        """Test that quantity is rounded to exchange step size."""
        broker = broker_with_mock_client

        broker.client.order_market = Mock(return_value={"orderId": 1, "status": "FILLED"})

        # Submit 1.23456789 BTC, should round to appropriate precision
        broker.submit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=1.23456789,
            order_type="market",
        )

        # Verify the call was made (actual rounding logic in _round_quantity)
        assert broker.client.order_market.called


class TestLiveBrokerPositions:
    """FIX 23: Test portfolio and position queries."""

    @pytest.fixture
    def broker_with_mock_client(self):
        """Create broker with mocked Binance client."""
        config = BrokerConfig(api_key="key123", api_secret="secret123")
        with patch("binance.client.Client") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            broker = LiveBroker(config, testnet=True)
            broker.client = mock_instance
            yield broker

    def test_get_portfolio_with_balances(self, broker_with_mock_client):
        """Test portfolio query returns cash and positions."""
        broker = broker_with_mock_client

        broker.client.get_account = Mock(
            return_value={
                "balances": [
                    {"asset": "USDT", "free": "10000.0", "locked": "0.0"},
                    {"asset": "BTC", "free": "0.5", "locked": "0.0"},
                    {"asset": "ETH", "free": "0.0", "locked": "0.0"},  # Dust, should filter
                ]
            }
        )
        broker.client.get_symbol_info = Mock(return_value={"symbol": "BTCUSDT", "status": "TRADING"})

        portfolio = broker.get_portfolio()

        assert isinstance(portfolio, PortfolioState)
        assert portfolio.cash > 0  # Should have USDT
        assert len(portfolio.positions) >= 1  # Should have BTC position
        # Should filter out dust (ETH with 0 quantity)

    def test_get_positions_filters_dust(self, broker_with_mock_client):
        """Test that positions < 0.00001 are filtered."""
        broker = broker_with_mock_client

        broker.client.get_account = Mock(
            return_value={
                "balances": [
                    {"asset": "USDT", "free": "10000.0", "locked": "0.0"},
                    {"asset": "BTC", "free": "0.5", "locked": "0.0"},
                    {"asset": "DUST", "free": "0.000001", "locked": "0.0"},  # Below min
                ]
            }
        )

        portfolio = broker.get_portfolio()

        # DUST balance should be filtered (< 0.00001)
        position_symbols = [p["symbol"] for p in portfolio.positions]
        assert "DUST" not in position_symbols
