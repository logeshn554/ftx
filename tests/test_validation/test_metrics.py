"""Tests for pure metric calculation functions."""

import numpy as np
import pytest
from trade.validation import metrics as m


def test_win_rate_and_profit_factor():
    pnls = [100.0, -50.0, 200.0, -50.0]
    assert m.win_rate(pnls) == 0.5
    assert m.profit_factor(pnls) == 300.0 / 100.0
    assert m.average_win(pnls) == 150.0
    assert m.average_loss(pnls) == 50.0
    assert m.expectancy(pnls) == 0.5 * 150.0 - 0.5 * 50.0


def test_cvar_tail_loss():
    returns = np.array([-0.10, -0.05, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
    cvar = m.cvar_tail_loss(returns, alpha=0.10)
    assert cvar >= 0.05


def test_market_exposure_pct():
    positions = [0.0, 1.5, 1.5, 0.0, -1.0]
    assert m.market_exposure_pct(positions) == 3 / 5


def test_cost_to_profit_ratio():
    assert m.cost_to_profit_ratio(10.0, 100.0) == 0.10
    assert m.cost_to_profit_ratio(10.0, 0.0) == float("inf")


def test_deflated_sharpe_ratio():
    # Single trial should return standard normal CDF
    dsr_single = m.deflated_sharpe_ratio(estimated_sharpe=1.5, num_trials=1, track_record_length=252)
    assert 0.0 <= dsr_single <= 1.0

    # With many trials (e.g. 1000 candidate searches), DSR penalizes false positives
    dsr_multiple = m.deflated_sharpe_ratio(estimated_sharpe=1.5, num_trials=1000, track_record_length=252)
    assert dsr_multiple < dsr_single
