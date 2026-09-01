"""Regression: champion policy must not mutate from trade outcomes or macro analysis."""

from scripts.webrl_engine import WebRLEngine


def test_macro_analysis_does_not_mutate_champion_policy():
    engine = WebRLEngine()
    before = engine.policy_adapter.base_policy.copy()
    before_conf = dict(before["pattern_confidences"])
    engine.total_trades = 100
    engine.run_100_attempt_macro_analysis()
    after = engine.policy_adapter.base_policy
    assert after["pattern_confidences"] == before_conf
    assert after["position_size_pct"] == before["position_size_pct"]
