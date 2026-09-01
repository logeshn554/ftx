"""Conservative expected-value gate for proposed trades."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpectedValue:
    p_win: float
    p_loss: float
    expected_win_return: float
    expected_loss_return: float
    total_expected_cost: float
    expected_value: float
    risk_reward_ratio: float
    confidence: float
    expected_move: float


class ExpectedValueFilter:
    def __init__(self, minimum_edge: float = 0.0, minimum_confidence: float = 0.55,
                 minimum_risk_reward: float = 1.0, cost_margin: float = 1.25):
        self.minimum_edge = minimum_edge
        self.minimum_confidence = minimum_confidence
        self.minimum_risk_reward = minimum_risk_reward
        self.cost_margin = cost_margin

    def estimate(self, p_win: float, expected_win_return: float, expected_loss_return: float,
                 expected_cost: float, confidence: float, expected_move: float | None = None) -> ExpectedValue:
        p_win = min(1.0, max(0.0, float(p_win)))
        p_loss = 1.0 - p_win
        win, loss, cost = float(expected_win_return), abs(float(expected_loss_return)), max(0.0, float(expected_cost))
        move = abs(float(expected_move if expected_move is not None else win))
        ev = p_win * win - p_loss * loss - cost
        return ExpectedValue(p_win, p_loss, win, loss, cost, ev, win / loss if loss else float("inf"), float(confidence), move)

    def validate(self, value: ExpectedValue) -> tuple[bool, str]:
        if value.confidence < self.minimum_confidence:
            return False, "insufficient_confidence"
        if value.risk_reward_ratio < self.minimum_risk_reward:
            return False, "risk_reward_below_minimum"
        if value.expected_move < value.total_expected_cost * self.cost_margin:
            return False, "expected_move_does_not_cover_cost"
        if value.expected_value <= self.minimum_edge:
            return False, "expected_value_below_minimum"
        return True, "accepted"

    def evaluate(self, **kwargs) -> tuple[ExpectedValue, bool, str]:
        value = self.estimate(**kwargs)
        accepted, reason = self.validate(value)
        return value, accepted, reason
