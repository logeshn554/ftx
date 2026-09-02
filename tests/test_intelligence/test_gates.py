from scripts.webrl_engine import QLearningSignalTable
from trade.intelligence.decision import DecisionPipeline


def test_decision_pipeline_holds_without_evidence():
    d = DecisionPipeline().decide("BUY", .9, .5, .01, .03, .002)
    assert d.action == "HOLD"
    assert d.reason in {"expected_value_below_minimum", "risk_reward_below_minimum", "no_gross_edge", "net_edge_below_minimum"}


def test_q_table_unseen_state_recommends_hold():
    q = QLearningSignalTable(min_trades=30)
    result = q.estimate_action({"trend": "FLAT"}, "BUY")
    assert result["recommendation"] == "HOLD"
    assert result["sample_count"] == 0
    assert result["uncertainty"] == 1.0


def test_expected_value_uncertainty_penalty():
    from trade.intelligence.expected_value import ExpectedValueFilter
    ev_filter = ExpectedValueFilter(cost_margin=1.0, uncertainty_penalty_weight=1.0)
    
    # Confident prediction with edge
    ev_high_conf, accepted_high, _ = ev_filter.evaluate(
        p_win=0.7,
        expected_win_return=0.03,
        expected_loss_return=0.01,
        expected_cost=0.003,
        confidence=0.9,
    )
    assert accepted_high
    assert ev_high_conf.trade_quality > 0

    # Highly uncertain prediction reduces edge below minimum
    ev_low_conf, accepted_low, reason = ev_filter.evaluate(
        p_win=0.7,
        expected_win_return=0.03,
        expected_loss_return=0.01,
        expected_cost=0.003,
        confidence=0.55,
        uncertainty=0.95,
    )
    assert not accepted_low or ev_low_conf.expected_value < ev_high_conf.expected_value
