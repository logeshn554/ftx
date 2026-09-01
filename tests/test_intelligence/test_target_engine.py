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


def test_tp_never_below_minimum():
    engine = TargetEngine(CostModel())
    plan = engine.plan(atr_pct=0.001, expected_move_pct=0.5)
    assert plan.take_profit_pct >= plan.minimum_target_pct
