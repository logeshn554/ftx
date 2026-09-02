"""Unit tests for PPOStrategy adapter."""

from unittest.mock import MagicMock
import numpy as np
import pytest
from trade.strategies.ppo_strategy import PPOStrategy


def test_ppo_strategy_abstains_when_no_model():
    strat = PPOStrategy(model=None)
    sig = strat.signal({"sma_10": 100.0, "sma_50": 95.0})
    assert sig.side == "HOLD"
    assert sig.reason == "ppo_model_not_loaded"


def test_ppo_strategy_emits_buy_on_action_1():
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.array([1]), None)

    strat = PPOStrategy(model=mock_model)
    sig = strat.signal({
        "sma_10": 100.0,
        "sma_50": 95.0,
        "adx": 25.0,
        "rsi_14": 55.0,
        "atr_pct": 1.5,
    })

    assert sig.side == "BUY"
    assert sig.strategy == "ppo"
    assert sig.confidence > 0.5
    assert sig.expected_move >= 1.5


def test_ppo_strategy_emits_sell_on_action_2():
    mock_model = MagicMock()
    mock_model.predict.return_value = (np.array([2]), None)

    strat = PPOStrategy(model=mock_model)
    sig = strat.signal({
        "sma_10": 90.0,
        "sma_50": 95.0,
        "adx": 25.0,
        "rsi_14": 40.0,
        "atr_pct": 1.5,
    })

    assert sig.side == "SELL"
    assert sig.strategy == "ppo"
    assert sig.confidence > 0.5
