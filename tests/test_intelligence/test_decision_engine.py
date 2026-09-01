from trade.intelligence.decision_engine import DecisionEngine


def test_decision_engine_holds_on_weak_signal():
    engine = DecisionEngine(minimum_signal_confidence=0.99)
    decision = engine.decide(
        indicators={"rsi_14": 50, "bb_pct": 0.5, "bb_width": 0},
        equity=10_000,
        entry_price=100,
    )
    assert decision.action == "HOLD"


def test_decision_engine_holds_when_cost_dominated():
    engine = DecisionEngine(cost_safety_multiplier=3.0, minimum_signal_confidence=0.5)
    decision = engine.decide(
        indicators={
            "rsi_14": 50, "bb_pct": 0.5, "bb_width": 0.001,
            "sma_10": 100, "sma_50": 100, "adx": 10,
            "atr_pct": 0.0001, "roc_10": 0.0,
        },
        equity=10_000,
        entry_price=100,
        regime_confidence=0.9,
    )
    assert decision.action == "HOLD"


def test_decision_result_has_versions():
    engine = DecisionEngine(model_version="champion-1", strategy_version="s-1")
    d = engine.decide(indicators={}, equity=1000, entry_price=100)
    assert d.model_version == "champion-1"
    assert d.strategy_version == "s-1"
