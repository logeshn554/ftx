"""Tests for the risk management engine."""

import pytest

from trade.core.types import Order, OrderSide, PortfolioState, Position
from trade.risk.engine import RiskEngine
from trade.risk.limits import RiskLimits
from trade.risk.circuit_breaker import CircuitBreaker, CircuitState


class TestRiskEngine:
    """Test the risk engine safety checks."""

    def test_approve_valid_order(self, sample_portfolio, sample_buy_order):
        """Valid order within all limits is approved."""
        engine = RiskEngine(RiskLimits())
        decision = engine.evaluate(
            sample_buy_order, sample_portfolio, current_price=300.0
        )
        assert decision.approved

    def test_reject_when_trading_disabled(self, sample_portfolio, sample_buy_order):
        """Orders rejected when trading is disabled."""
        engine = RiskEngine(RiskLimits())
        engine.disable_trading()

        decision = engine.evaluate(
            sample_buy_order, sample_portfolio, current_price=300.0
        )
        assert not decision.approved
        assert any("TRADING_DISABLED" in r for r in decision.rejections)

    def test_reject_daily_loss_exceeded(self, sample_portfolio, sample_buy_order):
        """Orders rejected when daily loss limit exceeded."""
        limits = RiskLimits(max_daily_loss_pct=5.0)
        engine = RiskEngine(limits)
        engine.reset_daily(100_000.0)
        engine.update_daily_pnl(-6000.0)  # 6% loss

        decision = engine.evaluate(
            sample_buy_order, sample_portfolio, current_price=300.0
        )
        assert not decision.approved
        assert any("DAILY_LOSS_LIMIT" in r for r in decision.rejections)

    def test_position_size_reduced(self, sample_portfolio):
        """Order quantity reduced when exceeding position size limit."""
        limits = RiskLimits(max_position_pct=10.0)  # 10% max
        engine = RiskEngine(limits)

        big_order = Order(symbol="TSLA", side=OrderSide.LONG, quantity=100)
        decision = engine.evaluate(
            big_order, sample_portfolio, current_price=500.0
        )
        # $50,000 order vs $9,550 limit (10% of $95,500)
        assert decision.approved
        assert decision.modified_order is not None
        assert decision.modified_order.quantity < big_order.quantity

    def test_reject_max_positions(self, sample_buy_order):
        """Reject new position when at max open positions."""
        limits = RiskLimits(max_open_positions=1)
        engine = RiskEngine(limits)

        portfolio = PortfolioState(
            cash=50_000.0,
            positions={"AAPL": Position("AAPL", 100, 150.0, 155.0)},
            total_equity=65_500.0,
        )

        decision = engine.evaluate(
            sample_buy_order, portfolio, current_price=300.0
        )
        assert not decision.approved
        assert any("MAX_POSITIONS" in r for r in decision.rejections)

    def test_reject_leverage_exceeded(self, sample_portfolio, sample_buy_order):
        """Reject when leverage would exceed limit."""
        limits = RiskLimits(max_leverage=1.0, max_position_pct=100.0)
        engine = RiskEngine(limits)

        # Portfolio already has positions worth ~$15,500
        # Adding another $15,000 order would push leverage above 1.0
        big_order = Order(symbol="TSLA", side=OrderSide.LONG, quantity=300)
        decision = engine.evaluate(
            big_order, sample_portfolio, current_price=300.0
        )
        assert not decision.approved
        assert any("LEVERAGE_EXCEEDED" in r for r in decision.rejections)

    def test_order_value_capped(self, sample_portfolio):
        """Order value is capped at max_order_value."""
        limits = RiskLimits(max_order_value=10_000.0, max_position_pct=100.0)
        engine = RiskEngine(limits)

        order = Order(symbol="TSLA", side=OrderSide.LONG, quantity=100)
        decision = engine.evaluate(
            order, sample_portfolio, current_price=500.0
        )
        # $50,000 order capped to $10,000 → 20 shares
        assert decision.approved
        assert decision.modified_order is not None
        assert decision.modified_order.quantity < 100

    def test_audit_log_recorded(self, sample_portfolio, sample_buy_order):
        """Every decision is recorded in the audit log."""
        engine = RiskEngine(RiskLimits())
        engine.evaluate(sample_buy_order, sample_portfolio, current_price=300.0)

        assert len(engine.audit_log) == 1
        assert engine.audit_log[0]["symbol"] == "MSFT"


class TestCircuitBreaker:
    """Test the circuit breaker mechanism."""

    def test_starts_closed(self):
        """Circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_trading_allowed

    def test_manual_trip(self):
        """Manual trip opens the circuit breaker."""
        cb = CircuitBreaker()
        cb.trip("Test trip")

        assert cb.state == CircuitState.OPEN
        assert not cb.is_trading_allowed

    def test_manual_reset(self):
        """Manual reset closes the circuit breaker."""
        cb = CircuitBreaker()
        cb.trip("Test")
        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.is_trading_allowed

    def test_consecutive_losses_trip(self):
        """Circuit breaker trips after consecutive losses."""
        cb = CircuitBreaker(max_consecutive_losses=3)

        cb.record_loss()
        cb.record_loss()
        assert cb.is_trading_allowed

        cb.record_loss()  # 3rd consecutive loss
        assert cb.state == CircuitState.OPEN

    def test_win_resets_consecutive(self):
        """A win resets the consecutive loss counter."""
        cb = CircuitBreaker(max_consecutive_losses=3)

        cb.record_loss()
        cb.record_loss()
        cb.record_win()  # resets counter
        cb.record_loss()

        assert cb.is_trading_allowed  # Only 1 consecutive loss

    def test_status_dict(self):
        """Status returns a complete dict."""
        cb = CircuitBreaker()
        status = cb.get_status()

        assert "state" in status
        assert "trading_allowed" in status
        assert "consecutive_losses" in status
        assert status["state"] == "CLOSED"
