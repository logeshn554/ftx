"""Unit tests for production Edge Gate and No-Trade scenarios."""

import pytest
from trade.intelligence.expected_value import ExpectedValueFilter


def test_scenario_a_obvious_profitable_edge():
    """Scenario A: Obvious profitable edge -> ACCEPT / TRADE."""
    gate = ExpectedValueFilter(minimum_edge=0.0, minimum_confidence=0.60)
    estimate, accepted, reason = gate.evaluate(
        p_win=0.70,
        expected_win_return=0.02,
        expected_loss_return=0.005,
        expected_cost=0.002,
        confidence=0.85,
        uncertainty=0.10,
    )
    assert accepted is True
    assert reason == "accepted"
    assert estimate.expected_net_edge > 0
    assert estimate.trade_quality > 1.0


def test_scenario_b_microscopic_edge_rejected():
    """Scenario B: Microscopic edge not covering friction -> REJECT / NO_TRADE."""
    gate = ExpectedValueFilter(minimum_edge=0.0, minimum_confidence=0.50)
    # Gross edge = 0.52 * 0.002 - 0.48 * 0.002 = 0.00008, cost = 0.0020
    estimate, accepted, reason = gate.evaluate(
        p_win=0.52,
        expected_win_return=0.002,
        expected_loss_return=0.002,
        expected_cost=0.002,
        confidence=0.60,
    )
    assert accepted is False
    assert reason in {"friction_not_covered", "net_edge_below_minimum"}
    assert estimate.expected_net_edge <= 0


def test_scenario_c_high_uncertainty_rejected():
    """Scenario C: High uncertainty dominates gross edge -> REJECT / NO_TRADE."""
    gate = ExpectedValueFilter(minimum_edge=0.0, uncertainty_penalty_weight=1.5)
    estimate, accepted, reason = gate.evaluate(
        p_win=0.60,
        expected_win_return=0.015,
        expected_loss_return=0.010,
        expected_cost=0.003,
        confidence=0.55,
        uncertainty=0.90,  # High outcome dispersion
    )
    assert accepted is False
    assert reason in {"uncertainty_dominates", "net_edge_below_minimum"}


def test_scenario_d_high_confidence_negative_ev_rejected():
    """Scenario D: High model confidence but negative EV -> REJECT / NO_TRADE."""
    gate = ExpectedValueFilter(minimum_edge=0.0)
    # Model is very confident (0.95), but market payoff is negative (win 0.001, loss 0.010, p_win 0.50)
    estimate, accepted, reason = gate.evaluate(
        p_win=0.50,
        expected_win_return=0.001,
        expected_loss_return=0.010,
        expected_cost=0.002,
        confidence=0.95,
    )
    assert accepted is False
    assert reason in {"no_gross_edge", "net_edge_below_minimum"}
