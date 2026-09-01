import pytest

from trade.execution.accounting import PositionAccounting


@pytest.mark.parametrize("side,entry,exit,positive", [
    ("BUY", 100.0, 110.0, True), ("BUY", 100.0, 90.0, False),
    ("SELL", 100.0, 90.0, True), ("SELL", 100.0, 110.0, False),
])
def test_realized_pnl_uses_actual_fills_for_long_and_short(side, entry, exit, positive):
    book = PositionAccounting(10_000, fee_pct=0, slippage_pct=0)
    assert book.open(side, 2, entry)
    trade = book.close(exit)
    assert trade is not None
    assert trade.gross_pnl == trade.net_pnl == (20 if positive else -20)
    assert trade.entry_price == entry and trade.exit_price == exit


def test_zero_movement_loses_fees_and_slippage_is_explicit():
    book = PositionAccounting(10_000, fee_pct=0.01, slippage_pct=0.01)
    book.open("BUY", 1, 100)
    trade = book.close(100)
    assert trade is not None
    assert trade.net_pnl < 0
    assert trade.entry_fee == pytest.approx(1.01)
    assert trade.exit_fee == pytest.approx(0.99)
    assert trade.slippage_cost == pytest.approx(2.0)


def test_partial_quantity_and_unrealized_market_mark():
    book = PositionAccounting(10_000, fee_pct=0, slippage_pct=0)
    book.open("SELL", 1.25, 100)
    assert book.quantity == 1.25
    assert book.unrealized_pnl(90) == pytest.approx(12.5)
    assert book.close(90).net_pnl == pytest.approx(12.5)
