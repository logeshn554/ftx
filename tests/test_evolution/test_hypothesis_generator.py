"""Unit tests for HypothesisGenerator."""

import datetime as dt
import pytest
from trade.evolution.experience_store import ImmutableExperienceStore
from trade.evolution.hypothesis_generator import HypothesisGenerator
from trade.experience.schema import TradeExperience


def test_hypothesis_generator_detects_regime_failure(tmp_path):
    store_file = tmp_path / "exp.json"
    store = ImmutableExperienceStore(path=store_file)

    now = dt.datetime.now(dt.timezone.utc)
    for i in range(15):
        store.append(
            TradeExperience(
                timestamp=now,
                symbol="BTCUSDT",
                timeframe="1h",
                market_features=(("adx", 12.0),),
                regime="SIDEWAYS",
                regime_confidence=0.8,
                strategy="trend",
                action="BUY",
                signal_confidence=0.6,
                expected_value=0.01,
                entry_price=100.0,
                exit_price=98.0,
                quantity=1.0,
                gross_pnl=-2.0,
                fees=0.2,
                slippage=0.05,
                net_pnl=-2.25,
                return_pct=-2.25,
                maximum_adverse_excursion=2.5,
                maximum_favorable_excursion=0.5,
                duration=5,
                drawdown_before=0.0,
                drawdown_after=0.02,
                outcome="LOSS",
                model_version="v1.0.0",
                strategy_version="v1",
                feature_version="v1",
            )
        )

    gen = HypothesisGenerator(min_sample_evidence=10)
    hypotheses = gen.generate_hypotheses(store, {"base_param": 1}, current_version="v1.0.0")

    assert len(hypotheses) >= 1
    regime_hyp = [h for h in hypotheses if h.hypothesis_type == "REGIME_MISMATCH"]
    assert len(regime_hyp) == 1
    assert "trend" in regime_hyp[0].premise
    assert "SIDEWAYS" in regime_hyp[0].premise
    assert regime_hyp[0].config["cost_safety_multiplier"] == 1.5


def test_hypothesis_generator_fallback_on_empty(tmp_path):
    store_file = tmp_path / "exp_empty.json"
    store = ImmutableExperienceStore(path=store_file)

    gen = HypothesisGenerator()
    hypotheses = gen.generate_hypotheses(store, {"reward_turnover_penalty": 0.05})
    assert len(hypotheses) == 1
    assert hypotheses[0].hypothesis_type == "EXPLORATION"
