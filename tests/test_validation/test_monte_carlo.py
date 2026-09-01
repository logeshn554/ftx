import numpy as np
import pytest

from trade.validation.monte_carlo import MonteCarloTester


def test_monte_carlo_positive_strategy_passes():
    # 50 winning trades of +$100, 20 losing trades of -$50
    pnls = [100.0] * 50 + [-50.0] * 20
    tester = MonteCarloTester(n_simulations=200, initial_capital=10_000.0, seed=42)
    res = tester.test(pnls)
    assert res.passed is True
    assert res.probability_of_ruin == 0.0
    assert res.positive_return_probability > 0.95


def test_monte_carlo_losing_strategy_fails():
    # 10 winning trades of +$20, 50 losing trades of -$200 (guaranteed high drawdown/ruin)
    pnls = [20.0] * 10 + [-200.0] * 50
    tester = MonteCarloTester(n_simulations=200, initial_capital=5_000.0, seed=42)
    res = tester.test(pnls)
    assert res.passed is False
    assert len(res.rejection_reasons) > 0


def test_monte_carlo_insufficient_trades():
    tester = MonteCarloTester(n_simulations=100)
    res = tester.test([10.0, -5.0, 2.0])
    assert res.passed is False
    assert "insufficient_trades" in res.rejection_reasons[0]
