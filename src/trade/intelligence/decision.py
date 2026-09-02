"""Canonical, auditable signal-to-trade decision pipeline with Edge Gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from .expected_value import EdgeEstimate, ExpectedValueFilter


@dataclass(frozen=True)
class Decision:
    action: str = "HOLD"
    reason: str = "insufficient_evidence"
    signal: str = "HOLD"
    confidence: float = 0.0
    expected_value: float = 0.0
    estimated_cost: float = 0.0
    risk: float = 0.0
    ev_detail: EdgeEstimate | None = None
    audit: dict[str, Any] = field(default_factory=dict)


class DecisionPipeline:
    def __init__(
        self,
        ev_filter: ExpectedValueFilter | None = None,
        minimum_signal_confidence: float = 0.55,
        maximum_risk: float = 0.35,
    ):
        self.ev_filter = ev_filter or ExpectedValueFilter()
        self.minimum_signal_confidence = minimum_signal_confidence
        self.maximum_risk = maximum_risk

    def decide(
        self,
        signal: str,
        confidence: float,
        p_win: float | None,
        expected_win_return: float,
        expected_loss_return: float,
        expected_cost: float,
        risk: float = 0.0,
        regime_valid: bool = True,
        execution_valid: bool = True,
        expected_move: float | None = None,
        uncertainty: float | None = None,
    ) -> Decision:
        signal = str(signal).upper()
        unc = float(uncertainty if uncertainty is not None else max(0.0, 1.0 - confidence))
        audit: dict[str, Any] = {
            "signal": signal,
            "confidence": confidence,
            "uncertainty": unc,
            "risk": risk,
            "estimated_cost": expected_cost,
        }
        
        if signal not in {"BUY", "SELL"}:
            return Decision(audit=audit, reason="no_trade_signal", signal=signal, confidence=confidence, risk=risk)
        if confidence < self.minimum_signal_confidence:
            return Decision(audit=audit, reason="low_confidence", signal=signal, confidence=confidence, risk=risk)
        if not regime_valid:
            return Decision(audit=audit, reason="regime_not_validated", signal=signal, confidence=confidence, risk=risk)
        if risk > self.maximum_risk:
            return Decision(audit=audit, reason="risk_limit_exceeded", signal=signal, confidence=confidence, risk=risk)
        if not execution_valid:
            return Decision(audit=audit, reason="data_quality_invalid", signal=signal, confidence=confidence, risk=risk)
        if p_win is None:
            return Decision(audit=audit, reason="missing_calibrated_probability", signal=signal, confidence=confidence, risk=risk)

        ev, accepted, reason = self.ev_filter.evaluate(
            p_win=p_win,
            expected_win_return=expected_win_return,
            expected_loss_return=expected_loss_return,
            expected_cost=expected_cost,
            confidence=confidence,
            expected_move=expected_move,
            uncertainty=unc,
        )
        audit.update({
            "p_win": ev.p_win,
            "expected_gross_return": ev.expected_gross_return,
            "expected_friction": ev.expected_friction,
            "expected_net_edge": ev.expected_net_edge,
            "uncertainty_margin": ev.uncertainty_margin,
            "trade_quality": ev.trade_quality,
            "risk_reward_ratio": ev.risk_reward_ratio,
            "rejection_reason": reason if not accepted else None,
        })
        return Decision(
            signal=signal,
            confidence=confidence,
            risk=risk,
            ev_detail=ev,
            action=signal if accepted else "HOLD",
            reason=reason,
            expected_value=ev.expected_net_edge,
            estimated_cost=expected_cost,
            audit=audit,
        )
