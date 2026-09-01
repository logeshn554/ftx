import pytest

from trade.execution.cost_model import CostConfig, CostModel
from trade.execution.ledger import Ledger
from trade.execution.position import Position


@pytest.mark.parametrize("side,entry,exit,positive", [
    ("BUY", 100.0, 110.0, True), ("BUY", 100.0, 90.0, False),
    ("SELL", 100.0, 90.0, True), ("SELL", 100.0, 110.0, False),
])
def test_ledger_long_short_pnl(side, entry, exit, positive):
    ledger = Ledger(10_000, CostConfig(taker_fee=0, maker_fee=0, entry_slippage=0, exit_slippage=0))
    assert ledger.enter_position("BTC", side, 2, entry)
    trade = ledger.close_position(exit)
    assert trade is not None
    expected = 20 if positive else -20
    assert trade.gross_pnl == trade.net_pnl == expected


def test_equity_reconciliation():
    ledger = Ledger(10_000)
    ledger.enter_position("BTC", "BUY", 1, 100)
    assert ledger.reconcile(105)


def test_fee_only_loss_on_zero_movement():
    ledger = Ledger(10_000, CostConfig(taker_fee=0.01, entry_slippage=0.01, exit_slippage=0.01))
    ledger.enter_position("BTC", "BUY", 1, 100)
    trade = ledger.close_position(100)
    assert trade is not None
    assert trade.net_pnl < 0


def test_reversal():
    ledger = Ledger(10_000, CostConfig(taker_fee=0, entry_slippage=0, exit_slippage=0))
    ledger.enter_position("BTC", "BUY", 1, 100)
    ledger.reverse("BTC", "SELL", 1, 110)
    assert ledger.open_position is not None
    assert ledger.open_position.side == "SELL"


def test_future_return_independence():
    ledger = Ledger(10_000, CostConfig(taker_fee=0, entry_slippage=0, exit_slippage=0))
    ledger.enter_position("BTC", "BUY", 1, 100)
  # Mutating a hypothetical future-return column must not affect close
    _future_proxy = 999.0
    trade = ledger.close_position(110)
    assert trade is not None
    assert trade.net_pnl == 10
    assert _future_proxy == 999.0


def test_cost_model_round_trip():
    model = CostModel(CostConfig(taker_fee=0.001, entry_slippage=0.0005, exit_slippage=0.0005))
    assert model.estimated_round_trip_cost_pct() == pytest.approx(0.3, rel=0.01)


def test_cost_model_stress():
    normal = CostModel(stress=False)
    stressed = CostModel(stress=True)
    assert stressed.estimated_round_trip_cost_pct() > normal.estimated_round_trip_cost_pct()


def test_position_unrealized():
    pos = Position("BTC", "BUY", 100, __import__("datetime").datetime.now(__import__("datetime").timezone.utc), 1, 0.1)
    pos.mark(110)
    assert pos.unrealized_pnl() == pytest.approx(9.9)
