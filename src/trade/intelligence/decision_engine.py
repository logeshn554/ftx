"""Canonical trading decision service — single pipeline for all runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trade.data.contract import observation_columns
from trade.execution.cost_model import CostConfig, CostModel
from trade.intelligence.decision import Decision, DecisionPipeline
from trade.intelligence.expected_value import ExpectedValueFilter
from trade.intelligence.strategy_selector import StrategySelector
from trade.intelligence.target_engine import TargetEngine
from trade.risk.cooldown import CooldownConfig, CooldownController
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
    """Canonical Trading Pipeline:
    Market Data -> Features -> Regime -> Strategies -> Strategy Selection -> EV Filter -> Cost Gate -> Risk & Survival -> Sizing -> Decision
    """

    def __init__(
        self,
        model_version: str = "champion-v0",
        strategy_version: str = "v1",
        minimum_signal_confidence: float = 0.55,
        maximum_risk: float = 0.35,
        cost_config: CostConfig | None = None,
        cost_safety_multiplier: float = 1.5,
        strategies: list[Strategy] | None = None,
        cooldown_config: CooldownConfig | None = None,
        strategy_selector: StrategySelector | None = None,
    ):
        self.model_version = model_version
        self.strategy_version = strategy_version
        self.cost_model = CostModel(cost_config or CostConfig())
        self.target_engine = TargetEngine(self.cost_model, cost_safety_multiplier=cost_safety_multiplier)
        self.ev_filter = ExpectedValueFilter(cost_margin=cost_safety_multiplier / 1.25)
        self.pipeline = DecisionPipeline(self.ev_filter, minimum_signal_confidence, maximum_risk)
        self.survival = SurvivalController()
        self.cooldown = CooldownController(cooldown_config)
        self.strategy_selector = strategy_selector or StrategySelector()
        self.strategies: list[Strategy] = strategies or [
            TrendStrategy(),
            MeanReversionStrategy(),
            MomentumStrategy(),
            BreakoutStrategy(),
        ]
        self._strategy_map: dict[str, Strategy] = {s.name: s for s in self.strategies}

    def _filter_indicators(self, indicators: dict) -> dict:
        """Enforce observation feature contract to guarantee no target/future leakage."""
        allowed = set(observation_columns(indicators.keys()))
        return {k: v for k, v in indicators.items() if k in allowed or k in {
            "trend", "rsi_zone", "bb_position", "momentum_20", "atr_pct", "atr_14", "volume_ratio"
        }}

    def _best_signal(
        self,
        indicators: dict,
        regime: str = "UNKNOWN",
        regime_performance: dict[str, dict] | None = None,
    ) -> Signal:
        clean_indicators = self._filter_indicators(indicators)

        # If empirical regime performance is available, select best strategy
        if regime_performance:
            selection = self.strategy_selector.select(regime, regime_performance)
            if selection.selected_strategy and selection.selected_strategy in self._strategy_map:
                sig = self._strategy_map[selection.selected_strategy].signal(clean_indicators)
                if sig.side in {"BUY", "SELL"} and sig.confidence > 0:
                    return sig

        # Evaluate candidate signals across all strategies
        candidates = [s.signal(clean_indicators) for s in self.strategies]
        actionable = [c for c in candidates if c.side in {"BUY", "SELL"} and c.confidence > 0]
        if not actionable:
            return Signal("HOLD", 0.0, 0.0, 0.0, 0.0, "none", "no_actionable_signal")
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
        daily_loss: float = 0.0,
        risk_score: float = 0.0,
        p_win: float | None = None,
        data_quality_ok: bool = True,
        drift_detected: bool = False,
        regime_performance: dict[str, dict] | None = None,
    ) -> TradeDecision:
        audit: dict[str, Any] = {"regime": regime, "regime_confidence": regime_confidence}

        # 1. Survival Controller Gate (Hard Halt)
        survival_state = self.survival.update(
            drawdown=drawdown,
            consecutive_losses=consecutive_losses,
            daily_loss=daily_loss,
            data_quality_ok=data_quality_ok,
            drift_detected=drift_detected,
        )
        audit["survival_state"] = survival_state.value
        if survival_state == SurvivalState.HALTED or not self.survival.allows_new_trade():
            reason = f"survival_{survival_state.value.lower()}" + (f"_{self.survival.halt_reason}" if self.survival.halt_reason else "")
            return self._hold(reason, audit)

        # 2. Anti-Churn & Cooldown Gate
        can_enter, cooldown_reason = self.cooldown.can_enter(equity)
        audit["cooldown_status"] = cooldown_reason
        if not can_enter:
            return self._hold(f"cooldown_blocked_{cooldown_reason.lower()}", audit)

        # 3. Regime Confidence Gate
        if regime_confidence < 0.4 and regime != "UNKNOWN":
            return self._hold("regime_confidence_insufficient", audit)

        # 4. Strategy Signal Generation
        signal = self._best_signal(indicators, regime=regime, regime_performance=regime_performance)
        audit["strategy"] = signal.strategy
        audit["signal_reason"] = signal.reason
        if signal.side == "HOLD":
            return self._hold("no_strategy_signal", audit)

        # 5. Target Engine & Cost Feasibility Gate
        atr_pct = float(indicators.get("atr_pct", indicators.get("atr_14", 0.0)))
        target_plan = self.target_engine.plan(
            atr_pct=atr_pct,
            expected_move_pct=signal.expected_move * 100 if signal.expected_move < 1 else signal.expected_move,
            strategy_target_pct=signal.target_distance * 100 if signal.target_distance < 1 else signal.target_distance,
        )
        audit["target_plan"] = target_plan.reason
        if not target_plan.should_trade:
            return self._hold(target_plan.reason, audit, signal)

        # 6. Expected Value (EV) Filter Gate
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

        # 7. Risk-Anchored Position Sizing
        stop_dist = entry_price * (target_plan.stop_loss_pct / 100)
        qty = position_size(
            equity=equity,
            entry_price=entry_price,
            stop_distance=stop_dist,
            edge=max(0.0, ev_decision.expected_value),
            confidence=signal.confidence,
            volatility=atr_pct / 100 if atr_pct else 0.01,
            drawdown=drawdown,
            reward_to_risk=target_plan.take_profit_pct / max(target_plan.stop_loss_pct, 0.01),
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
