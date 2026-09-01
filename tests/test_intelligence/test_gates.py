from scripts.webrl_engine import QLearningSignalTable
from trade.intelligence.decision import DecisionPipeline


def test_decision_pipeline_holds_without_evidence():
    d = DecisionPipeline().decide("BUY", .9, .5, .01, .03, .002)
    assert d.action == "HOLD"
    assert d.reason in {"expected_value_below_minimum", "risk_reward_below_minimum"}


def test_q_table_unseen_state_recommends_hold():
    q = QLearningSignalTable(min_trades=30)
    result = q.estimate_action({"trend": "FLAT"}, "BUY")
    assert result["recommendation"] == "HOLD"
    assert result["sample_count"] == 0
    assert result["uncertainty"] == 1.0
