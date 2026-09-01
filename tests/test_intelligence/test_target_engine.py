from trade.execution.cost_model import CostModel
from trade.intelligence.target_engine import TargetEngine


def test_cost_dominated_trade_rejected():
    engine = TargetEngine(CostModel(), cost_safety_multiplier=1.5)
    plan = engine.plan(atr_pct=0.01, expected_move_pct=0.02)
    assert plan.should_trade is False
    assert plan.reason in {"EXPECTED_MOVE_BELOW_COST", "TARGET_BELOW_MINIMUM"}


def test_viable_trade_accepted():
    engine = TargetEngine(CostModel(), cost_safety_multiplier=1.25)
    plan = engine.plan(atr_pct=0.5, expected_move_pct=1.0)
    assert plan.should_trade is True
    assert plan.take_profit_pct >= plan.minimum_target_pct


def test_tp_not_inflated_to_cover_costs():
    """Bug #2 fix: TP must never be enlarged just to cover costs.
    When ATR is tiny but expected move is large, the strategy TP
    (from tiny ATR) should be rejected — not inflated."""
    engine = TargetEngine(CostModel())
    plan = engine.plan(atr_pct=0.001, expected_move_pct=0.5)
    # ATR * 1.5 = 0.0015 → well below cost minimum, so rejected
    assert plan.should_trade is False
    assert plan.reason == "TARGET_BELOW_MINIMUM"


def test_expected_move_below_cost_is_hold():
    """If expected move cannot cover round-trip cost × safety, answer is HOLD."""
    engine = TargetEngine(CostModel(), cost_safety_multiplier=2.0)
    plan = engine.plan(atr_pct=0.5, expected_move_pct=0.01)
    assert plan.should_trade is False
    assert plan.reason == "EXPECTED_MOVE_BELOW_COST"


def test_strategy_target_used_directly():
    """When strategy provides its own target, use it — don't inflate to cost floor."""
    engine = TargetEngine(CostModel(), cost_safety_multiplier=1.25)
    plan = engine.plan(atr_pct=0.5, expected_move_pct=1.0, strategy_target_pct=0.8)
    assert plan.should_trade is True
    assert plan.take_profit_pct == 0.8  # Strategy target used directly


def test_risk_reward_insufficient_rejected():
    """If computed R:R is below min_risk_reward, reject."""
    engine = TargetEngine(CostModel(), min_risk_reward=3.0, atr_multiplier_tp=0.5, atr_multiplier_sl=2.0)
    plan = engine.plan(atr_pct=1.0, expected_move_pct=1.0)
    # TP = 0.5, SL = max(2.0, 0.5/3.0) = 2.0, R:R = 0.5/2.0 = 0.25 < 3.0
    assert plan.should_trade is False
    assert plan.reason == "RISK_REWARD_INSUFFICIENT"
