import pytest

from trade.risk.cooldown import CooldownConfig, CooldownController
from trade.risk.position_sizing import position_size
from trade.risk.survival import SurvivalController, SurvivalState


def test_survival_controller_hard_halts():
    sc = SurvivalController(caution_drawdown=0.05, defensive_drawdown=0.10, halt_drawdown=0.20, max_daily_loss=0.05)
    
    # Normal
    assert sc.update(drawdown=0.02) == SurvivalState.NORMAL
    assert sc.allows_new_trade() is True
    
    # Caution
    assert sc.update(drawdown=0.06) == SurvivalState.CAUTION
    assert sc.allows_new_trade() is True
    
    # Defensive
    assert sc.update(drawdown=0.12) == SurvivalState.DEFENSIVE
    assert sc.allows_new_trade() is False
    
    # Halt on max drawdown
    assert sc.update(drawdown=0.21) == SurvivalState.HALTED
    assert sc.allows_new_trade() is False
    
    # Halt on model degradation
    sc.recover(force=True)
    assert sc.update(drawdown=0.01, model_degradation=True) == SurvivalState.HALTED
    
    # Halt on daily loss breach
    sc.recover(force=True)
    assert sc.update(drawdown=0.01, daily_loss=0.06) == SurvivalState.HALTED


def test_position_sizing_anchored_on_stop():
    # 10k equity, 1% risk budget = $100. Stop distance $50 -> 2.0 units max 1R budget.
    # With edge=0.10, b=1.0: Half-Kelly = 0.5 * (0.10 / 1.0) * 10000 = $500 notional -> 5 units.
    # So 1R risk budget (2 units) is the binding constraint.
    qty = position_size(
        equity=10_000,
        entry_price=100,
        stop_distance=50,
        edge=0.10,
        confidence=1.0,
        volatility=0.0,
        drawdown=0.0,
        max_risk_per_trade=0.01,
        max_position_pct=0.20,
        reward_to_risk=1.0,
    )
    assert qty == pytest.approx(2.0)
    
    # Drawdown reduces position size linearly
    qty_dd = position_size(
        equity=10_000,
        entry_price=100,
        stop_distance=50,
        edge=0.10,
        confidence=1.0,
        volatility=0.0,
        drawdown=0.10,
        max_risk_per_trade=0.01,
        max_position_pct=0.20,
        reward_to_risk=1.0,
    )
    assert qty_dd < qty
    assert qty_dd == pytest.approx(1.8)


def test_cooldown_anti_churn():
    config = CooldownConfig(min_inter_trade_bars=2, min_inter_trade_seconds=0.0, max_trades_per_window=3, window_bars=10)
    ctrl = CooldownController(config)
    
    # Initial: can enter
    allowed, _ = ctrl.can_enter(10_000)
    assert allowed is True
    
    # Enter at bar 5
    ctrl.record_entry(notional=1000, bar_index=5)
    
    # Bar 5 immediately after: blocked by inter-trade cooldown
    ctrl.advance_bar(5)
    allowed, reason = ctrl.can_enter(10_000)
    assert allowed is False
    assert "INTER_TRADE_COOLDOWN" in reason
    
    # Bar 6: waited 1 bar, still blocked
    ctrl.advance_bar(6)
    allowed, _ = ctrl.can_enter(10_000)
    assert allowed is False
    
    # Bar 7: waited 2 bars, allowed
    ctrl.advance_bar(7)
    allowed, _ = ctrl.can_enter(10_000)
    assert allowed is True
