"""Canonical trading decision service with Edge Gate, Regime Engine, Calibrated Probability and Uncertainty."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from trade.data.contract import observation_columns
from trade.execution.cost_model import CostConfig, CostModel
from trade.intelligence.decision import Decision, DecisionPipeline
from trade.intelligence.expected_value import ExpectedValueFilter
from trade.intelligence.probability_calibrator import CalibratedProbabilityEstimator
from trade.intelligence.regime import MarketRegime, RegimeClassification, RegimeEngine
from trade.intelligence.strategy_selector import StrategySelector
from trade.intelligence.target_engine import TargetEngine
from trade.intelligence.uncertainty import UncertaintyEstimate, UncertaintyEstimator
from trade.risk.cooldown import CooldownConfig, CooldownController
from trade.risk.limits import RiskLimits
from trade.risk.position_sizing import position_size
from trade.risk.survival import SurvivalController, SurvivalState
from trade.strategies.base import Signal, Strategy
from trade.strategies.breakout import BreakoutStrategy
from trade.strategies.mean_reversion import MeanReversionStrategy
from trade.strategies.momentum import MomentumStrategy
from trade.strategies.trend import TrendStrategy


@dataclass(frozen=True)
class TradeDecision:
    action: str  # TRADE | HOLD | FORCE_FLAT | REJECT
    side: str | None

    confidence: float
    uncertainty: float

    expected_return: float
    expected_gross_return: float
    expected_cost: float
    expected_net_edge: float

    trade_quality: float

    risk: float
    target_pct: float
    stop_pct: float
    position_size: float

    strategy: str | None
    reason: str

    model_version: str
    strategy_version: str

    audit: dict[str, Any] = field(default_factory=dict)

    # Backward compatibility properties
    @property
    def expected_value(self) -> float:
        return self.expected_net_edge

    @property
    def estimated_cost(self) -> float:
        return self.expected_cost

    @property
    def expected_move(self) -> float:
        return self.expected_return


class DecisionEngine:
    """Canonical Trading Pipeline:
    Market Data -> Features -> Regime Engine -> Strategies -> Strategy Selection -> Edge Gate -> Grounded Uncertainty -> Risk Sizing -> Decision
    """

    def __init__(
        self,
        model_version: str = "champion-v0",
        strategy_version: str = "v1",
        minimum_signal_confidence: float = 0.55,
        maximum_risk: float = 0.35,
        cost_config: CostConfig | None = None,
        cost_safety_multiplier: float = 1.0,
        strategies: list[Strategy] | None = None,
        cooldown_config: CooldownConfig | None = None,
        strategy_selector: StrategySelector | None = None,
        regime_engine: RegimeEngine | None = None,
        probability_calibrator: CalibratedProbabilityEstimator | None = None,
        uncertainty_estimator: UncertaintyEstimator | None = None,
        risk_limits: RiskLimits | None = None,
    ):
        self.model_version = model_version
        self.strategy_version = strategy_version
        self.cost_model = CostModel(cost_config or CostConfig())
        self.target_engine = TargetEngine(self.cost_model, cost_safety_multiplier=cost_safety_multiplier)
        self.ev_filter = ExpectedValueFilter(cost_margin=cost_safety_multiplier)
        self.pipeline = DecisionPipeline(self.ev_filter, minimum_signal_confidence, maximum_risk)
        self.survival = SurvivalController()
        self.cooldown = CooldownController(cooldown_config)
        self.strategy_selector = strategy_selector or StrategySelector()
        self.regime_engine = regime_engine or RegimeEngine()
        self.probability_calibrator = probability_calibrator or CalibratedProbabilityEstimator()
        self.uncertainty_estimator = uncertainty_estimator or UncertaintyEstimator()
        self.risk_limits = risk_limits or RiskLimits()
        self.strategies: list[Strategy] = strategies or [
            TrendStrategy(),
            MeanReversionStrategy(),
            MomentumStrategy(),
            BreakoutStrategy(),
        ]
        self._strategy_map: dict[str, Strategy] = {s.name: s for s in self.strategies}
        self.rejected_trades: list[dict[str, Any]] = []

    def _filter_indicators(self, indicators: dict) -> dict:
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

        if regime_performance:
            selection = self.strategy_selector.select(regime, regime_performance)
            if selection.selected_strategy and selection.selected_strategy in self._strategy_map:
                sig = self._strategy_map[selection.selected_strategy].signal(clean_indicators)
                if sig.side in {"BUY", "SELL"} and sig.confidence > 0:
                    return sig

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
        uncertainty: float | None = None,
    ) -> TradeDecision:
        # Automatic Regime Detection if unassigned
        if regime == "UNKNOWN" or regime_confidence <= 0.0:
            reg_class = self.regime_engine.classify(indicators)
            regime = reg_class.name
            regime_confidence = reg_class.confidence

        atr_pct = float(indicators.get("atr_pct", indicators.get("atr_14", 1.0)))

        # Automatic Grounded Uncertainty Estimation if not provided
        if uncertainty is None:
            feat_vec = [
                float(indicators.get("adx", 15.0)),
                float(indicators.get("rsi_14", 50.0)),
                atr_pct,
                float(indicators.get("bb_width_pct", 2.0)),
            ]
            unc_est = self.uncertainty_estimator.estimate(feat_vec, current_volatility_pct=atr_pct)
            uncertainty = unc_est.total_uncertainty

        audit: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": entry_price,
            "equity": equity,
            "regime": regime,
            "regime_confidence": regime_confidence,
            "epistemic_uncertainty": uncertainty,
        }

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

        # 3. Panic Crash Regime Circuit Breaker
        if regime == MarketRegime.PANIC_CRASH.value:
            return self._hold("panic_crash_regime_circuit_breaker", audit)

        # 4. Regime Confidence Gate
        if regime_confidence < 0.4 and regime != "UNKNOWN":
            return self._hold("regime_confidence_insufficient", audit)

        # 5. Strategy Signal Generation
        signal = self._best_signal(indicators, regime=regime, regime_performance=regime_performance)
        audit["strategy"] = signal.strategy
        audit["signal_reason"] = signal.reason
        if signal.side == "HOLD":
            return self._hold("no_strategy_signal", audit)

        # 6. Target Engine & Cost Feasibility Gate
        target_plan = self.target_engine.plan(
            atr_pct=atr_pct,
            expected_move_pct=signal.expected_move,
            strategy_target_pct=signal.target_distance,
        )
        audit["target_plan"] = target_plan.reason
        if not target_plan.should_trade:
            return self._hold(target_plan.reason, audit, signal)

        # 7. Calibrated Probability Prediction
        calibrated_p_win = p_win
        if calibrated_p_win is None:
            feat_arr = np.array([
                float(indicators.get("adx", 15.0)),
                float(indicators.get("rsi_14", 50.0)),
                atr_pct,
                float(indicators.get("momentum_20", 0.0)),
            ])
            calibrated_p_win = self.probability_calibrator.predict_p_win(feat_arr)

        if calibrated_p_win is None and hasattr(signal, "probability"):
            calibrated_p_win = getattr(signal, "probability")

        est_cost_frac = self.cost_model.estimated_round_trip_cost_fraction()
        ev_decision: Decision = self.pipeline.decide(
            signal=signal.side,
            confidence=signal.confidence,
            p_win=calibrated_p_win,
            expected_win_return=target_plan.take_profit_pct / 100,
            expected_loss_return=target_plan.stop_loss_pct / 100,
            expected_cost=est_cost_frac,
            risk=risk_score,
            regime_valid=regime_confidence >= 0.4 or regime == "UNKNOWN",
            execution_valid=data_quality_ok,
            expected_move=target_plan.expected_move_pct / 100,
            uncertainty=uncertainty,
        )
        audit.update(ev_decision.audit)
        if ev_decision.action == "HOLD":
            gross_ret = ev_decision.ev_detail.expected_gross_return if ev_decision.ev_detail else 0.0
            net_edge = ev_decision.expected_value
            trade_qual = ev_decision.ev_detail.trade_quality if ev_decision.ev_detail else 0.0
            unc = ev_decision.ev_detail.uncertainty if ev_decision.ev_detail else (uncertainty or max(0.0, 1.0 - signal.confidence))
            return self._hold(
                ev_decision.reason,
                audit,
                signal,
                expected_gross_return=gross_ret,
                expected_net_edge=net_edge,
                trade_quality=trade_qual,
                uncertainty=unc,
            )

        # 8. Risk-Anchored Position Sizing
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
            risk_limits=self.risk_limits,
        )
        if qty <= 0:
            return self._hold("position_size_zero", audit, signal)

        ev_detail = ev_decision.ev_detail
        gross_ret = ev_detail.expected_gross_return if ev_detail else 0.0
        net_edge = ev_detail.expected_net_edge if ev_detail else 0.0
        trade_qual = ev_detail.trade_quality if ev_detail else 0.0
        unc = ev_detail.uncertainty if ev_detail else (uncertainty or max(0.0, 1.0 - signal.confidence))

        return TradeDecision(
            action="TRADE",
            side=signal.side,
            confidence=signal.confidence,
            uncertainty=unc,
            expected_return=target_plan.expected_move_pct,
            expected_gross_return=gross_ret * 100,
            expected_cost=est_cost_frac * 100,
            expected_net_edge=net_edge * 100,
            trade_quality=trade_qual,
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
        expected_gross_return: float = 0.0,
        expected_net_edge: float = 0.0,
        trade_quality: float = 0.0,
        uncertainty: float = 0.0,
    ) -> TradeDecision:
        conf = signal.confidence if signal else 0.0
        unc = uncertainty if uncertainty > 0 else max(0.0, 1.0 - conf)
        cost_pct = self.cost_model.estimated_round_trip_cost_pct()

        rejection_record = {
            "timestamp": audit.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "proposed_side": signal.side if signal else None,
            "strategy": signal.strategy if signal else None,
            "confidence": conf,
            "uncertainty": unc,
            "expected_gross_return": expected_gross_return,
            "expected_cost": cost_pct,
            "expected_net_edge": expected_net_edge,
            "trade_quality": trade_quality,
            "reason": reason,
        }
        self.rejected_trades.append(rejection_record)
        audit["rejection_logged"] = True

        return TradeDecision(
            action="HOLD",
            side=None,
            confidence=conf,
            uncertainty=unc,
            expected_return=signal.expected_move if signal else 0.0,
            expected_gross_return=expected_gross_return,
            expected_cost=cost_pct,
            expected_net_edge=expected_net_edge,
            trade_quality=trade_quality,
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
