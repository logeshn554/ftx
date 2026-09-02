"""Production-Grade Edge Gate and Uncertainty No-Trade Engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgeEstimate:
    """Rich economic edge estimate for a proposed trade."""
    p_win: float
    p_loss: float
    expected_win_return: float
    expected_loss_return: float
    expected_gross_return: float
    expected_friction: float
    uncertainty: float
    uncertainty_margin: float
    expected_net_edge: float
    trade_quality: float
    confidence: float
    expected_move: float

    # Aliases for backward compatibility
    @property
    def expected_value(self) -> float:
        return self.expected_net_edge

    @property
    def total_expected_cost(self) -> float:
        return self.expected_friction

    @property
    def risk_reward_ratio(self) -> float:
        return self.expected_win_return / self.expected_loss_return if self.expected_loss_return > 0 else float("inf")


# Maintain alias for ExpectedValue
ExpectedValue = EdgeEstimate


class ExpectedValueFilter:
    """Rigorous No-Trade gate evaluating net edge after friction and uncertainty."""

    def __init__(
        self,
        minimum_edge: float = 0.0,
        minimum_confidence: float = 0.55,
        minimum_risk_reward: float = 1.0,
        cost_margin: float = 1.0,
        uncertainty_penalty_weight: float = 0.5,
        minimum_trade_quality: float = 0.0,
    ):
        self.minimum_edge = float(minimum_edge)
        self.minimum_confidence = float(minimum_confidence)
        self.minimum_risk_reward = float(minimum_risk_reward)
        self.cost_margin = float(cost_margin)
        self.uncertainty_penalty_weight = float(uncertainty_penalty_weight)
        self.minimum_trade_quality = float(minimum_trade_quality)

    def estimate(
        self,
        p_win: float,
        expected_win_return: float,
        expected_loss_return: float,
        expected_cost: float,
        confidence: float,
        expected_move: float | None = None,
        uncertainty: float | None = None,
    ) -> EdgeEstimate:
        p_w = min(1.0, max(0.0, float(p_win)))
        p_l = 1.0 - p_w
        win = max(0.0, float(expected_win_return))
        loss = max(0.0, abs(float(expected_loss_return)))
        friction = max(0.0, float(expected_cost))
        conf = min(1.0, max(0.0, float(confidence)))
        unc = float(uncertainty if uncertainty is not None else max(0.0, 1.0 - conf))
        unc_margin = unc * self.uncertainty_penalty_weight * friction
        move = abs(float(expected_move if expected_move is not None else win))

        # Expected Gross Return = P(win) * E[win] - P(loss) * E[loss]
        gross_return = p_w * win - p_l * loss
        
        # Expected Net Edge = Expected Gross Return - Expected Friction - Uncertainty Margin
        net_edge = gross_return - friction - unc_margin
        
        # Trade Quality = Expected Net Edge / (Expected Friction + Uncertainty Margin)
        denominator = max(1e-8, friction + unc_margin)
        trade_quality = net_edge / denominator

        return EdgeEstimate(
            p_win=p_w,
            p_loss=p_l,
            expected_win_return=win,
            expected_loss_return=loss,
            expected_gross_return=gross_return,
            expected_friction=friction,
            uncertainty=unc,
            uncertainty_margin=unc_margin,
            expected_net_edge=net_edge,
            trade_quality=trade_quality,
            confidence=conf,
            expected_move=move,
        )

    def validate(self, estimate: EdgeEstimate) -> tuple[bool, str]:
        """Evaluate trade through the production No-Trade gate hierarchy."""
        if estimate.confidence < self.minimum_confidence:
            return False, "low_confidence"
        if estimate.expected_gross_return <= 0:
            return False, "no_gross_edge"
        if estimate.expected_gross_return <= estimate.expected_friction:
            return False, "friction_not_covered"
        if estimate.expected_move < estimate.expected_friction * self.cost_margin:
            return False, "expected_move_does_not_cover_cost"
        if estimate.uncertainty_margin >= estimate.expected_gross_return:
            return False, "uncertainty_dominates"
        if estimate.expected_net_edge <= self.minimum_edge:
            return False, "net_edge_below_minimum"
        if estimate.risk_reward_ratio < self.minimum_risk_reward:
            return False, "risk_reward_below_minimum"
        if estimate.trade_quality < self.minimum_trade_quality:
            return False, "trade_quality_below_threshold"
        return True, "accepted"

    def evaluate(self, **kwargs) -> tuple[EdgeEstimate, bool, str]:
        estimate = self.estimate(**kwargs)
        accepted, reason = self.validate(estimate)
        return estimate, accepted, reason
