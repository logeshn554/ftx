"""Canonical trading decision service — single pipeline for all runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trade.execution.cost_model import CostConfig, CostModel
from trade.intelligence.decision import Decision, DecisionPipeline
from trade.intelligence.expected_value import ExpectedValueFilter
from trade.intelligence.target_engine import TargetEngine
from trade.risk.position_sizing import position_size
from trade.risk.survival import SurvivalController, SurvivalState
from trade.strategies.base import Signal, Strategy
from trade.strategies.breakout import BreakoutStrategy
from trade.strategies.mean_reversion import MeanReversionStrategy
from trade.strategies.momentum import MomentumStrategy
from trade.strategies.trend import TrendStrategy


@dataclass(frozen=True)
class TradeDecision:
    action: str  # TRADE | HOLD
    side: str | None
    confidence: float
    expected_value: float
    expected_move: float
    estimated_cost: float
    risk: float
    target_pct: float
    stop_pct: float
    position_size: float
    strategy: str | None
    reason: str
    model_version: str
    strategy_version: str
    audit: dict[str, Any] = field(default_factory=dict)


class DecisionEngine:
    """market data → validation → features → regime → strategies → EV → cost → risk → sizing → decision"""

    def __init__(
        self,
        model_version: str = "champion-v0",
        strategy_version: str = "v1",
        minimum_signal_confidence: float = 0.55,
        maximum_risk: float = 0.35,
        cost_config: CostConfig | None = None,
        cost_safety_multiplier: float = 1.5,
        strategies: list[Strategy] | None = None,
    ):
        self.model_version = model_version
        self.strategy_version = strategy_version
        self.cost_model = CostModel(cost_config or CostConfig())
        self.target_engine = TargetEngine(self.cost_model, cost_safety_multiplier=cost_safety_multiplier)
        self.ev_filter = ExpectedValueFilter(cost_margin=cost_safety_multiplier / 1.25)
        self.pipeline = DecisionPipeline(self.ev_filter, minimum_signal_confidence, maximum_risk)
        self.survival = SurvivalController()
        self.strategies: list[Strategy] = strategies or [
            TrendStrategy(),
            MeanReversionStrategy(),
            MomentumStrategy(),
            BreakoutStrategy(),
        ]

    def _best_signal(self, indicators: dict) -> Signal:
        candidates = [s.signal(indicators) for s in self.strategies]
        actionable = [c for c in candidates if c.side in {"BUY", "SELL"} and c.confidence > 0]
        if not actionable:
            return Signal("HOLD", 0.0, 0.0, 0.0, 0.0, "none")
        return max(actionable, key=lambda s: s.confidence)

    def decide(
        self,
        indicators: dict,
        equity: float,
        entry_price: float,
        regime: str = "UNKNOWN",
        regime_confidence: float = 0.0,
        drawdown: float = 0.0,
        consecutive_losses: int = 0,
        risk_score: float = 0.0,
        p_win: float | None = None,
        data_quality_ok: bool = True,
    ) -> TradeDecision:
        audit: dict[str, Any] = {"regime": regime, "regime_confidence": regime_confidence}
        survival_state = self.survival.update(drawdown, consecutive_losses, data_quality_ok=data_quality_ok)
        audit["survival_state"] = survival_state.value

        if survival_state == SurvivalState.HALTED or not self.survival.allows_new_trade():
            return self._hold("survival_halted", audit)

        if regime_confidence < 0.4 and regime != "UNKNOWN":
            return self._hold("regime_confidence_insufficient", audit)

        signal = self._best_signal(indicators)
        audit["strategy"] = signal.strategy
        if signal.side == "HOLD":
            return self._hold("no_strategy_signal", audit)

        atr_pct = float(indicators.get("atr_pct", indicators.get("atr_14", 0.0)))
        target_plan = self.target_engine.plan(
            atr_pct=atr_pct,
            expected_move_pct=signal.expected_move * 100 if signal.expected_move < 1 else signal.expected_move,
            strategy_target_pct=signal.target_distance * 100 if signal.target_distance < 1 else signal.target_distance,
        )
        audit["target_plan"] = target_plan.reason
        if not target_plan.should_trade:
            return self._hold(target_plan.reason, audit, signal)

        est_cost_frac = self.cost_model.estimated_round_trip_cost_fraction()
        win_p = p_win if p_win is not None else min(1.0, 0.5 + signal.confidence * 0.3)
        ev_decision: Decision = self.pipeline.decide(
            signal=signal.side,
            confidence=signal.confidence,
            p_win=win_p,
            expected_win_return=target_plan.take_profit_pct / 100,
            expected_loss_return=target_plan.stop_loss_pct / 100,
            expected_cost=est_cost_frac,
            risk=risk_score,
            regime_valid=regime_confidence >= 0.4 or regime == "UNKNOWN",
            execution_valid=data_quality_ok,
            expected_move=target_plan.expected_move_pct / 100,
        )
        audit.update(ev_decision.audit)
        if ev_decision.action == "HOLD":
            return self._hold(ev_decision.reason, audit, signal, ev_decision.expected_value)

        qty = position_size(
            equity=equity,
            entry_price=entry_price,
            stop_distance=entry_price * target_plan.stop_loss_pct / 100,
            edge=max(0.0, ev_decision.expected_value),
            confidence=signal.confidence,
            volatility=atr_pct / 100 if atr_pct else 0.01,
            drawdown=drawdown,
        )
        if qty <= 0:
            return self._hold("position_size_zero", audit, signal, ev_decision.expected_value)

        return TradeDecision(
            action="TRADE",
            side=signal.side,
            confidence=signal.confidence,
            expected_value=ev_decision.expected_value,
            expected_move=target_plan.expected_move_pct,
            estimated_cost=est_cost_frac * 100,
            risk=risk_score,
            target_pct=target_plan.take_profit_pct,
            stop_pct=target_plan.stop_loss_pct,
            position_size=qty,
            strategy=signal.strategy,
            reason="accepted",
            model_version=self.model_version,
            strategy_version=self.strategy_version,
            audit=audit,
        )

    def _hold(
        self,
        reason: str,
        audit: dict,
        signal: Signal | None = None,
        ev: float = 0.0,
    ) -> TradeDecision:
        return TradeDecision(
            action="HOLD",
            side=None,
            confidence=signal.confidence if signal else 0.0,
            expected_value=ev,
            expected_move=signal.expected_move if signal else 0.0,
            estimated_cost=self.cost_model.estimated_round_trip_cost_pct(),
            risk=0.0,
            target_pct=0.0,
            stop_pct=0.0,
            position_size=0.0,
            strategy=signal.strategy if signal else None,
            reason=reason,
            model_version=self.model_version,
            strategy_version=self.strategy_version,
            audit=audit,
        )
