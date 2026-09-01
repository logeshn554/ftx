"""Cost-aware take-profit and stop-loss target computation."""

from __future__ import annotations

from dataclasses import dataclass

from trade.execution.cost_model import CostModel


@dataclass(frozen=True)
class TargetPlan:
    should_trade: bool
    take_profit_pct: float
    stop_loss_pct: float
    minimum_target_pct: float
    expected_move_pct: float
    round_trip_cost_pct: float
    reason: str


class TargetEngine:
    """Ensure favorable movement exceeds transaction cost plus safety margin."""

    def __init__(
        self,
        cost_model: CostModel | None = None,
        cost_safety_multiplier: float = 1.5,
        min_risk_reward: float = 1.2,
        atr_multiplier_tp: float = 1.5,
        atr_multiplier_sl: float = 1.2,
    ):
        self.cost_model = cost_model or CostModel()
        self.cost_safety_multiplier = cost_safety_multiplier
        self.min_risk_reward = min_risk_reward
        self.atr_multiplier_tp = atr_multiplier_tp
        self.atr_multiplier_sl = atr_multiplier_sl

    def plan(
        self,
        atr_pct: float,
        expected_move_pct: float,
        strategy_target_pct: float | None = None,
    ) -> TargetPlan:
        round_trip = self.cost_model.estimated_round_trip_cost_pct()
        minimum_target = round_trip * self.cost_safety_multiplier
        atr_move = max(0.0, float(atr_pct))
        proposed_tp = strategy_target_pct if strategy_target_pct is not None else atr_move * self.atr_multiplier_tp
        proposed_tp = max(proposed_tp, minimum_target)
        proposed_sl = max(atr_move * self.atr_multiplier_sl, minimum_target / self.min_risk_reward)
        move = max(expected_move_pct, proposed_tp)

        if move <= minimum_target:
            return TargetPlan(
                should_trade=False,
                take_profit_pct=proposed_tp,
                stop_loss_pct=proposed_sl,
                minimum_target_pct=minimum_target,
                expected_move_pct=expected_move_pct,
                round_trip_cost_pct=round_trip,
                reason="EXPECTED_MOVE_BELOW_COST",
            )
        if proposed_tp <= minimum_target:
            return TargetPlan(
                should_trade=False,
                take_profit_pct=proposed_tp,
                stop_loss_pct=proposed_sl,
                minimum_target_pct=minimum_target,
                expected_move_pct=expected_move_pct,
                round_trip_cost_pct=round_trip,
                reason="TARGET_BELOW_MINIMUM",
            )
        return TargetPlan(
            should_trade=True,
            take_profit_pct=proposed_tp,
            stop_loss_pct=proposed_sl,
            minimum_target_pct=minimum_target,
            expected_move_pct=expected_move_pct,
            round_trip_cost_pct=round_trip,
            reason="ACCEPTED",
        )
