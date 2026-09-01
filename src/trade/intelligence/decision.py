"""One canonical, auditable signal-to-trade decision pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from .expected_value import ExpectedValue, ExpectedValueFilter


@dataclass(frozen=True)
class Decision:
    action: str = "HOLD"
    reason: str = "insufficient_evidence"
    signal: str = "HOLD"
    confidence: float = 0.0
    expected_value: float = 0.0
    estimated_cost: float = 0.0
    risk: float = 0.0
    ev_detail: ExpectedValue | None = None
    audit: dict = field(default_factory=dict)


class DecisionPipeline:
    def __init__(self, ev_filter: ExpectedValueFilter | None = None, minimum_signal_confidence: float = 0.55,
                 maximum_risk: float = 0.35):
        self.ev_filter = ev_filter or ExpectedValueFilter()
        self.minimum_signal_confidence = minimum_signal_confidence
        self.maximum_risk = maximum_risk

    def decide(self, signal: str, confidence: float, p_win: float, expected_win_return: float,
               expected_loss_return: float, expected_cost: float, risk: float = 0.0,
               regime_valid: bool = True, execution_valid: bool = True,
               expected_move: float | None = None) -> Decision:
        signal = str(signal).upper()
        audit = {"signal": signal, "confidence": confidence, "risk": risk, "estimated_cost": expected_cost}
        if signal not in {"BUY", "SELL"}:
            return Decision(audit=audit, reason="no_trade_signal", signal=signal, confidence=confidence, risk=risk)
        if confidence < self.minimum_signal_confidence:
            return Decision(audit=audit, reason="signal_confidence_below_minimum", signal=signal, confidence=confidence, risk=risk)
        if not regime_valid:
            return Decision(audit=audit, reason="regime_not_validated", signal=signal, confidence=confidence, risk=risk)
        if risk > self.maximum_risk:
            return Decision(audit=audit, reason="risk_limit_exceeded", signal=signal, confidence=confidence, risk=risk)
        if not execution_valid:
            return Decision(audit=audit, reason="execution_not_validated", signal=signal, confidence=confidence, risk=risk)
        ev, accepted, reason = self.ev_filter.evaluate(p_win=p_win, expected_win_return=expected_win_return,
            expected_loss_return=expected_loss_return, expected_cost=expected_cost, confidence=confidence, expected_move=expected_move)
        audit = {**audit, "expected_value": ev.expected_value, "risk_reward_ratio": ev.risk_reward_ratio}
        return Decision(signal=signal, confidence=confidence, risk=risk, ev_detail=ev,
                        action=signal if accepted else "HOLD", reason=reason,
                        expected_value=ev.expected_value, estimated_cost=expected_cost, audit=audit)
