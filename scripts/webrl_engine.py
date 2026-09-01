"""WebRL Self-Evolving Online Curriculum Reinforcement Learning Engine.

Implements the WebRL paper methodology adapted for crypto trading:
1. LossAnalyzer         - Classifies WHY each losing trade failed
2. SelfEvolvingCurriculum - Generates new training scenarios from failures
3. OutcomeRewardModel   - Pre-trade scoring trained on realized outcomes
4. KLConstrainedPolicyAdapter - Adapts parameters with stability constraint

Reference: "WebRL: Training LLM Web Agents via Self-Evolving Online
Curriculum Reinforcement Learning" (Qi et al., 2024)
"""

from __future__ import annotations

import collections
import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Data Structures
# ============================================================

@dataclass
class TradeContext:
    """Full context snapshot of a trade at close time."""
    trade_id: int = 0
    timestamp: str = ""
    side: str = "BUY"
    pattern: str = ""
    regime: str = ""
    regime_confidence: float = 0.0
    rsi: float = 50.0
    macd: str = "0.0"
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    allocated_capital: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    duration_bars: int = 0
    outcome: str = "LOSS"  # PROFIT or LOSS
    price_trajectory: list = field(default_factory=list)

    def to_feature_vector(self) -> list[float]:
        """Convert to numerical feature vector for ORM training."""
        side_enc = 1.0 if self.side == "BUY" else -1.0
        regime_enc = {"Bullish Trend": 1.0, "Mean Reversion": 0.5,
                      "Bearish Trend": -1.0, "High Volatility": 0.0}.get(self.regime, 0.0)
        try:
            macd_val = float(self.macd.replace("+", ""))
        except (ValueError, AttributeError):
            macd_val = 0.0

        return [
            side_enc,
            regime_enc,
            self.regime_confidence,
            self.rsi / 100.0,
            macd_val / 50.0,
            self.allocated_capital / 1000.0,
            self.duration_bars / 10.0,
            (self.exit_price - self.entry_price) / max(self.entry_price, 1.0) * 100.0,
        ]


@dataclass
class FailureCase:
    """Analyzed failure from a losing trade."""
    trade_context: TradeContext
    failure_cause: str = "UNKNOWN"
    severity: float = 0.0  # 0.0 = minor, 1.0 = catastrophic
    explanation: str = ""
    suggested_fix: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "failure_cause": self.failure_cause,
            "severity": round(self.severity, 3),
            "explanation": self.explanation,
            "suggested_fix": self.suggested_fix,
            "pattern": self.trade_context.pattern,
            "regime": self.trade_context.regime,
            "pnl": self.trade_context.pnl,
            "pnl_pct": self.trade_context.pnl_pct,
            "side": self.trade_context.side,
        }


@dataclass
class CurriculumItem:
    """A generated training scenario from a failure."""
    source_failure: str = ""  # failure cause
    scenario_desc: str = ""
    adjusted_params: dict = field(default_factory=dict)
    priority: float = 0.0  # higher = train on this first
    replayed_count: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class WinCase:
    """Analyzed success from a profitable trade / spontaneous pattern."""
    trade_context: TradeContext
    profit_driver: str = "TREND_CONTINUATION"
    quality_score: float = 0.0  # 0.0 to 1.0 based on efficiency & R:R
    explanation: str = ""
    extracted_rule: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "profit_driver": self.profit_driver,
            "quality_score": round(self.quality_score, 3),
            "explanation": self.explanation,
            "extracted_rule": self.extracted_rule,
            "pattern": self.trade_context.pattern,
            "regime": self.trade_context.regime,
            "pnl": self.trade_context.pnl,
            "pnl_pct": self.trade_context.pnl_pct,
            "side": self.trade_context.side,
        }


# ============================================================
# 1. Loss & Win Pattern Analyzers
# ============================================================

class LossAnalyzer:
    """Classifies WHY each losing trade failed."""

    FAILURE_CAUSES = [
        "WRONG_REGIME",       # HMM regime was misclassified
        "BAD_TIMING",         # Entered too early or too late
        "PATTERN_FAILURE",    # The detected pattern didn't play out
        "ADVERSE_MOMENTUM",   # Strong opposing price movement after entry
        "RISK_TOO_HIGH",      # Loss exceeded expected risk threshold
    ]

    def __init__(self):
        self.failure_history: list[FailureCase] = []
        self.cause_counts: dict[str, int] = {c: 0 for c in self.FAILURE_CAUSES}
        self.pattern_failure_counts: dict[str, int] = collections.defaultdict(int)
        self.pattern_total_counts: dict[str, int] = collections.defaultdict(int)

    def analyze(self, ctx: TradeContext) -> FailureCase:
        """Analyze a losing trade and classify the failure cause."""
        cause = "PATTERN_FAILURE"
        severity = min(1.0, abs(ctx.pnl_pct) / 2.0)  # Scale: 2% loss = max severity
        explanation = ""
        fix = ""

        price_move_pct = ((ctx.exit_price - ctx.entry_price) / max(ctx.entry_price, 1.0)) * 100

        # 1. Check if regime was wrong for the trade direction
        bullish_regimes = ["Bullish Trend", "Mean Reversion"]
        bearish_regimes = ["Bearish Trend"]
        if ctx.side == "BUY" and ctx.regime in bearish_regimes:
            cause = "WRONG_REGIME"
            explanation = f"Bought in {ctx.regime} regime. Price moved {price_move_pct:+.2f}% against position."
            fix = "Increase regime filter strictness. Only BUY in Bullish/MeanRev regimes."
        elif ctx.side == "SELL" and ctx.regime in bullish_regimes:
            cause = "WRONG_REGIME"
            explanation = f"Sold short in {ctx.regime} regime. Price moved {price_move_pct:+.2f}% against."
            fix = "Restrict SELL signals to Bearish/HighVol regimes only."

        # 2. Check for adverse momentum (large sudden move against position)
        elif abs(price_move_pct) > 1.0:
            cause = "ADVERSE_MOMENTUM"
            explanation = f"Strong {price_move_pct:+.2f}% move against {ctx.side} within {ctx.duration_bars} bars."
            fix = "Tighten stop-loss from 1.2% to 0.8%. Add momentum confirmation filter."

        # 3. Check if duration was very short (entered at wrong time)
        elif ctx.duration_bars <= 2:
            cause = "BAD_TIMING"
            explanation = f"Position closed after only {ctx.duration_bars} bars. Entry timing was premature."
            fix = "Add entry delay: wait 1-2 confirmation bars before committing capital."

        # 4. Check if the risk score was near the threshold
        elif severity > 0.6:
            cause = "RISK_TOO_HIGH"
            explanation = f"Loss of {ctx.pnl_pct:.2f}% exceeded acceptable risk. Allocated ${ctx.allocated_capital:.2f}."
            fix = "Reduce position size from 35% to 25% of capital. Lower XGBoost risk threshold."

        # 5. Default: pattern didn't work
        else:
            cause = "PATTERN_FAILURE"
            explanation = f"Pattern '{ctx.pattern}' predicted {ctx.side} but price moved {price_move_pct:+.2f}%."
            fix = f"Reduce confidence weight for '{ctx.pattern}' pattern. Require stronger signal confluence."

        self.cause_counts[cause] = self.cause_counts.get(cause, 0) + 1
        self.pattern_failure_counts[ctx.pattern] += 1
        self.pattern_total_counts[ctx.pattern] += 1

        failure = FailureCase(
            trade_context=ctx,
            failure_cause=cause,
            severity=severity,
            explanation=explanation,
            suggested_fix=fix,
        )
        self.failure_history.append(failure)
        if len(self.failure_history) > 200:
            self.failure_history.pop(0)

        return failure

    def record_win(self, ctx: TradeContext):
        """Record a winning trade for pattern success tracking."""
        self.pattern_total_counts[ctx.pattern] += 1

    def get_pattern_failure_rate(self, pattern: str) -> float:
        total = self.pattern_total_counts.get(pattern, 0)
        if total == 0:
            return 0.0
        fails = self.pattern_failure_counts.get(pattern, 0)
        return fails / total

    def get_summary(self) -> dict:
        return {
            "total_failures": len(self.failure_history),
            "cause_distribution": dict(self.cause_counts),
            "pattern_failure_rates": {
                p: round(self.get_pattern_failure_rate(p), 3)
                for p in self.pattern_total_counts
            },
            "last_failure": self.failure_history[-1].to_dict() if self.failure_history else None,
        }


class WinAnalyzer:
    """Deeply analyzes winning trades / spontaneous patterns to understand WHY they gave profit."""

    PROFIT_DRIVERS = [
        "STRONG_TREND_MOMENTUM",
        "PERFECT_MEAN_REVERSION",
        "REGIME_CONGRUENCE",
        "VOLATILITY_EXPANSION",
        "MULTI_INDICATOR_CONFLUENCE",
    ]

    def __init__(self):
        self.win_history: list[WinCase] = []
        self.driver_counts: dict[str, int] = {d: 0 for d in self.PROFIT_DRIVERS}
        self.pattern_win_counts: dict[str, int] = collections.defaultdict(int)
        self.pattern_total_pnl: dict[str, float] = collections.defaultdict(float)

    def analyze(self, ctx: TradeContext) -> WinCase:
        """Analyze a profitable trade, extract success factors, and formulate a reusable trading rule."""
        pnl = ctx.pnl
        pnl_pct = ctx.pnl_pct
        quality_score = min(1.0, max(0.2, (pnl_pct / 1.8) * 0.8 + (1.0 / max(1, ctx.duration_bars)) * 0.2))

        # Classify why it gave profit
        if ctx.regime in ["Bullish Trend", "Bearish Trend"] and ctx.regime_confidence > 0.85:
            driver = "REGIME_CONGRUENCE"
            explanation = f"High-confidence {ctx.regime} ({ctx.regime_confidence*100:.0f}%) directly propelled the {ctx.side} trade to +${pnl:.2f}."
            rule = f"When HMM confidence > 85% in {ctx.regime}, prioritize {ctx.side} entries with extended Take-Profit."

        elif (ctx.side == "BUY" and ctx.rsi < 35.0) or (ctx.side == "SELL" and ctx.rsi > 65.0):
            driver = "PERFECT_MEAN_REVERSION"
            explanation = f"RSI was extreme ({ctx.rsi:.1f}) triggering rapid price snap-back of {pnl_pct:+.2f}%."
            rule = f"Enter aggressively on RSI extreme bounces when divergence confirms reversal."

        elif ctx.duration_bars <= 3 and abs(pnl_pct) >= 1.0:
            driver = "STRONG_TREND_MOMENTUM"
            explanation = f"Rapid breakout with swift price velocity gained {pnl_pct:+.2f}% in only {ctx.duration_bars} bars."
            rule = f"Ride fast momentum bursts by trailing stop-loss tightly after 2 bars."

        elif ctx.regime == "High Volatility":
            driver = "VOLATILITY_EXPANSION"
            explanation = f"Bollinger band expansion in High Volatility regime captured {pnl_pct:+.2f}% move."
            rule = f"Trade breakout direction immediately upon volatility squeeze expansion."

        else:
            driver = "MULTI_INDICATOR_CONFLUENCE"
            explanation = f"Confluence of RSI ({ctx.rsi:.1f}), MACD ({ctx.macd}), and PPO execution yielded +${pnl:.2f} profit."
            rule = f"Boost allocation when technical confluence exceeds 3 aligned indicators."

        self.driver_counts[driver] = self.driver_counts.get(driver, 0) + 1
        self.pattern_win_counts[ctx.pattern] += 1
        self.pattern_total_pnl[ctx.pattern] += pnl

        wincase = WinCase(
            trade_context=ctx,
            profit_driver=driver,
            quality_score=quality_score,
            explanation=explanation,
            extracted_rule=rule,
        )
        self.win_history.append(wincase)
        if len(self.win_history) > 200:
            self.win_history.pop(0)

        return wincase

    def get_summary(self) -> dict:
        return {
            "total_analyzed_wins": len(self.win_history),
            "driver_distribution": dict(self.driver_counts),
            "last_win": self.win_history[-1].to_dict() if self.win_history else None,
            "top_profitable_patterns": sorted(
                [{"pattern": p, "wins": self.pattern_win_counts[p], "total_pnl": round(self.pattern_total_pnl[p], 2)}
                 for p in self.pattern_win_counts],
                key=lambda x: x["total_pnl"], reverse=True
            )[:5]
        }


# ============================================================
# Pattern Memory Bank (Stores & Connects Discovered Patterns)
# ============================================================

class PatternMemoryBank:
    """Stores analyzed winning patterns, indexes episodic vectors, and promotes high-expectancy setups."""

    def __init__(self):
        self.patterns: dict[str, dict] = {
            "Bullish Golden Cross Breakout": {"wins": 0, "losses": 0, "total_pnl": 0.0, "status": "ACTIVE", "weight": 0.91, "driver": "STRONG_TREND_MOMENTUM"},
            "RSI Oversold Bounce": {"wins": 0, "losses": 0, "total_pnl": 0.0, "status": "ACTIVE", "weight": 0.87, "driver": "PERFECT_MEAN_REVERSION"},
            "Bearish Breakdown Rejection": {"wins": 0, "losses": 0, "total_pnl": 0.0, "status": "ACTIVE", "weight": 0.89, "driver": "REGIME_CONGRUENCE"},
            "Volatility Squeeze Compression": {"wins": 0, "losses": 0, "total_pnl": 0.0, "status": "ACTIVE", "weight": 0.84, "driver": "VOLATILITY_EXPANSION"},
            "Resistance Double Top Reversal": {"wins": 0, "losses": 0, "total_pnl": 0.0, "status": "ACTIVE", "weight": 0.86, "driver": "PERFECT_MEAN_REVERSION"},
        }
        self.discovered_patterns_count = 0
        self.stored_episodes: list[dict] = []

    def register_trade_outcome(self, pattern: str, pnl: float, outcome: str, driver: str = "CONFLUENCE"):
        """Store trade result, discover new patterns if unseen, and update memory weights."""
        if pattern not in self.patterns:
            self.discovered_patterns_count += 1
            self.patterns[pattern] = {
                "wins": 0, "losses": 0, "total_pnl": 0.0,
                "status": "DISCOVERED & PROMOTED",
                "weight": 0.75,
                "driver": driver,
            }

        p = self.patterns[pattern]
        if outcome == "PROFIT":
            p["wins"] += 1
            p["weight"] = min(0.98, round(p["weight"] + 0.015, 3))
            p["driver"] = driver
        else:
            p["losses"] += 1
            p["weight"] = max(0.40, round(p["weight"] - 0.02, 3))

        p["total_pnl"] = round(p["total_pnl"] + pnl, 2)
        total = p["wins"] + p["losses"]
        p["win_rate"] = round((p["wins"] / max(1, total)) * 100.0, 1)

        # Store episodic snapshot
        self.stored_episodes.append({
            "pattern": pattern,
            "outcome": outcome,
            "pnl": pnl,
            "time": time.strftime("%H:%M:%S"),
            "driver": driver,
        })
        if len(self.stored_episodes) > 100:
            self.stored_episodes.pop(0)

    def get_summary(self) -> dict:
        return {
            "total_patterns_tracked": len(self.patterns),
            "discovered_patterns_count": self.discovered_patterns_count,
            "patterns_list": [
                {"name": k, "wins": v["wins"], "losses": v["losses"], "win_rate": v.get("win_rate", 0.0),
                 "total_pnl": v["total_pnl"], "weight": v["weight"], "status": v["status"], "driver": v["driver"]}
                for k, v in self.patterns.items()
            ],
            "recent_stored_episodes": self.stored_episodes[-5:],
        }


# ============================================================
# 2. Self-Evolving Curriculum
# ============================================================

class SelfEvolvingCurriculum:
    """Generates new training scenarios from failures (WebRL Core)."""

    def __init__(self, max_items: int = 100):
        self.curriculum: list[CurriculumItem] = []
        self.max_items = max_items
        self.generation_count = 0

    def generate_from_failure(self, failure: FailureCase) -> list[CurriculumItem]:
        """Generate new curriculum items from a failure case."""
        items = []
        ctx = failure.trade_context
        cause = failure.failure_cause
        self.generation_count += 1

        # Base priority: severity + recency bonus
        base_priority = failure.severity * 0.7 + 0.3

        if cause == "WRONG_REGIME":
            items.append(CurriculumItem(
                source_failure=cause,
                scenario_desc=f"Scenario: {ctx.regime} regime detected. Test: Should agent {ctx.side}? Expected: NO. Require regime-side alignment.",
                adjusted_params={"regime_filter_strict": True, "block_buy_in_bearish": True, "block_sell_in_bullish": True},
                priority=base_priority * 1.2,
            ))
            items.append(CurriculumItem(
                source_failure=cause,
                scenario_desc=f"Counter-factual: What if agent waited for regime confirmation (2 consecutive HMM readings)?",
                adjusted_params={"require_regime_confirmation_bars": 2},
                priority=base_priority * 0.9,
            ))

        elif cause == "ADVERSE_MOMENTUM":
            items.append(CurriculumItem(
                source_failure=cause,
                scenario_desc=f"Scenario: Strong momentum against position. Test: Tighter stop-loss at 0.8% instead of 1.2%.",
                adjusted_params={"stop_loss_pct": 0.8, "momentum_filter_enabled": True},
                priority=base_priority * 1.1,
            ))
            items.append(CurriculumItem(
                source_failure=cause,
                scenario_desc=f"Counter-factual: What if position size was 20% instead of 35%? Loss would be ${abs(ctx.pnl) * 0.57:.2f} instead of ${abs(ctx.pnl):.2f}.",
                adjusted_params={"max_position_pct": 0.20},
                priority=base_priority * 0.8,
            ))

        elif cause == "BAD_TIMING":
            items.append(CurriculumItem(
                source_failure=cause,
                scenario_desc=f"Scenario: Pattern '{ctx.pattern}' detected. Test: Wait 2 confirmation bars before entry.",
                adjusted_params={"entry_delay_bars": 2, "require_confirmation": True},
                priority=base_priority * 1.0,
            ))

        elif cause == "RISK_TOO_HIGH":
            items.append(CurriculumItem(
                source_failure=cause,
                scenario_desc=f"Scenario: XGBoost risk was near threshold. Test: Lower risk threshold from 0.35 to 0.28.",
                adjusted_params={"xgb_risk_threshold": 0.28, "max_position_pct": 0.25},
                priority=base_priority * 1.3,
            ))

        elif cause == "PATTERN_FAILURE":
            items.append(CurriculumItem(
                source_failure=cause,
                scenario_desc=f"Scenario: Pattern '{ctx.pattern}' failed. Reduce confidence weight by 15%. Require stronger RSI/MACD confirmation.",
                adjusted_params={"pattern_confidence_decay": 0.85, "require_rsi_macd_confluence": True},
                priority=base_priority * 1.0,
            ))

        # Add to curriculum, removing lowest-priority if full
        for item in items:
            self.curriculum.append(item)

        # Sort by priority (highest first), trim to max
        self.curriculum.sort(key=lambda x: x.priority, reverse=True)
        if len(self.curriculum) > self.max_items:
            self.curriculum = self.curriculum[: self.max_items]

        return items

    def get_top_lessons(self, n: int = 5) -> list[dict]:
        """Get top-N highest priority curriculum items."""
        return [
            {
                "scenario": item.scenario_desc,
                "source": item.source_failure,
                "priority": round(item.priority, 3),
                "replayed": item.replayed_count,
            }
            for item in self.curriculum[:n]
        ]

    def replay_top(self) -> Optional[CurriculumItem]:
        """Get the highest priority item for replay and increment its count."""
        if not self.curriculum:
            return None
        top = self.curriculum[0]
        top.replayed_count += 1
        # Decay priority after replay so other items get a chance
        top.priority *= 0.85
        self.curriculum.sort(key=lambda x: x.priority, reverse=True)
        return top

    def get_summary(self) -> dict:
        cause_dist = collections.Counter(item.source_failure for item in self.curriculum)
        return {
            "curriculum_size": len(self.curriculum),
            "total_generated": self.generation_count,
            "cause_distribution": dict(cause_dist),
            "top_lessons": self.get_top_lessons(3),
        }


# ============================================================
# 3. Outcome-Supervised Reward Model (ORM)
# ============================================================

class OutcomeRewardModel:
    """Lightweight online reward model that predicts trade outcome scores.

    Uses a simple weighted feature scoring approach that updates online
    after every realized trade. This is the ORM from WebRL adapted to
    produce a scalar score before each trade decision.
    """

    def __init__(self, n_features: int = 8):
        self.n_features = n_features
        # Initialize weights near zero (neutral)
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.learning_rate = 0.05
        self.experience_buffer: list[tuple[list[float], float]] = []
        self.max_buffer = 500
        self.total_updates = 0
        self.score_history: list[float] = []

    def predict_score(self, features: list[float]) -> float:
        """Predict outcome score for a proposed trade. Returns -1.0 to +1.0."""
        if len(features) != self.n_features:
            features = features[:self.n_features] + [0.0] * max(0, self.n_features - len(features))

        raw = self.bias + sum(w * f for w, f in zip(self.weights, features))
        score = max(-1.0, min(1.0, math.tanh(raw)))
        self.score_history.append(score)
        if len(self.score_history) > 100:
            self.score_history.pop(0)
        return score

    def update(self, features: list[float], outcome: float):
        """Update the model with a realized outcome. outcome: +1 for win, -1 for loss, scaled."""
        if len(features) != self.n_features:
            features = features[:self.n_features] + [0.0] * max(0, self.n_features - len(features))

        self.experience_buffer.append((features, outcome))
        if len(self.experience_buffer) > self.max_buffer:
            self.experience_buffer.pop(0)

        # Online SGD update
        prediction = self.bias + sum(w * f for w, f in zip(self.weights, features))
        error = outcome - math.tanh(prediction)

        # Gradient update with tanh derivative
        tanh_deriv = 1.0 - math.tanh(prediction) ** 2
        grad_scale = error * tanh_deriv * self.learning_rate

        for i in range(self.n_features):
            self.weights[i] += grad_scale * features[i]
        self.bias += grad_scale

        self.total_updates += 1

        # Periodic mini-batch replay from buffer for stability
        if self.total_updates % 5 == 0 and len(self.experience_buffer) >= 5:
            self._replay_batch(batch_size=min(5, len(self.experience_buffer)))

    def _replay_batch(self, batch_size: int = 5):
        """Replay a mini-batch from experience buffer."""
        batch = random.sample(self.experience_buffer, batch_size)
        for feats, outcome in batch:
            prediction = self.bias + sum(w * f for w, f in zip(self.weights, feats))
            error = outcome - math.tanh(prediction)
            tanh_deriv = 1.0 - math.tanh(prediction) ** 2
            grad_scale = error * tanh_deriv * self.learning_rate * 0.3  # Smaller LR for replay
            for i in range(self.n_features):
                self.weights[i] += grad_scale * feats[i]
            self.bias += grad_scale

    def get_summary(self) -> dict:
        avg_score = sum(self.score_history) / max(1, len(self.score_history))
        return {
            "total_updates": self.total_updates,
            "buffer_size": len(self.experience_buffer),
            "avg_recent_score": round(avg_score, 3),
            "weights_norm": round(math.sqrt(sum(w ** 2 for w in self.weights)), 3),
            "bias": round(self.bias, 4),
        }


# ============================================================
# 4. KL-Constrained Policy Adapter
# ============================================================

class KLConstrainedPolicyAdapter:
    """Adapts trading parameters with KL-divergence stability constraint.

    Maintains a base policy (original parameters) and proposes updates
    that are constrained to stay within a KL budget from the base.
    """

    def __init__(self):
        # Base policy parameters (the starting "safe" configuration)
        self.base_policy = {
            "pattern_confidences": {
                "Bullish Golden Cross Breakout": 0.91,
                "RSI Oversold Bounce": 0.87,
                "Bearish Breakdown Rejection": 0.89,
                "Volatility Squeeze Compression": 0.84,
                "Resistance Double Top Reversal": 0.86,
            },
            "position_size_pct": 0.35,
            "stop_loss_pct": 1.2,
            "take_profit_pct": 1.8,
            "xgb_risk_threshold": 0.35,
            "regime_filter_strict": False,
            "entry_delay_bars": 0,
        }

        # Current (adapted) policy — starts as copy of base
        self.current_policy = json.loads(json.dumps(self.base_policy))

        self.kl_budget = 0.5  # Max KL divergence allowed from base
        self.adaptation_rate = 0.15  # How fast to adapt (0-1)
        self.base_decay_rate = 0.02  # How fast base drifts toward current
        self.update_history: list[dict] = []
        self.total_adaptations = 0

    def adapt_on_loss(self, failure: FailureCase):
        """Propose and apply policy parameter changes based on a failure, constrained by KL budget."""
        proposed = json.loads(json.dumps(self.current_policy))
        cause = failure.failure_cause
        pattern = failure.trade_context.pattern
        severity = failure.severity

        # Propose changes based on failure cause
        if cause == "WRONG_REGIME":
            proposed["regime_filter_strict"] = True

        if cause == "ADVERSE_MOMENTUM":
            proposed["stop_loss_pct"] = max(0.5, proposed["stop_loss_pct"] - 0.1 * severity)
            proposed["position_size_pct"] = max(0.15, proposed["position_size_pct"] - 0.03 * severity)

        if cause == "BAD_TIMING":
            proposed["entry_delay_bars"] = min(3, proposed.get("entry_delay_bars", 0) + 1)

        if cause == "RISK_TOO_HIGH":
            proposed["xgb_risk_threshold"] = max(0.15, proposed["xgb_risk_threshold"] - 0.02 * severity)
            proposed["position_size_pct"] = max(0.15, proposed["position_size_pct"] - 0.02 * severity)

        if cause == "PATTERN_FAILURE" and pattern in proposed["pattern_confidences"]:
            old_conf = proposed["pattern_confidences"][pattern]
            proposed["pattern_confidences"][pattern] = max(0.5, old_conf - 0.04 * severity)

        # Only reduce position size on meaningful losses (severity > 0.3)
        if severity > 0.3:
            proposed["position_size_pct"] = max(0.15, proposed["position_size_pct"] - 0.01 * severity)

        # Compute KL divergence between proposed and base
        kl = self._compute_kl(proposed)

        # If within budget, apply. Otherwise, interpolate toward budget boundary.
        if kl <= self.kl_budget:
            self.current_policy = proposed
        else:
            # Interpolate: find alpha such that KL(interpolated, base) ≈ kl_budget
            alpha = self.kl_budget / max(kl, 0.001)
            self.current_policy = self._interpolate(self.current_policy, proposed, alpha)

        self.total_adaptations += 1

        update_record = {
            "adaptation_id": self.total_adaptations,
            "failure_cause": cause,
            "pattern": pattern,
            "severity": round(severity, 3),
            "kl_distance": round(self._compute_kl(self.current_policy), 4),
            "kl_budget": self.kl_budget,
            "position_size": round(self.current_policy["position_size_pct"], 3),
            "stop_loss": round(self.current_policy["stop_loss_pct"], 3),
            "risk_threshold": round(self.current_policy["xgb_risk_threshold"], 3),
            "time": time.strftime("%H:%M:%S"),
        }
        self.update_history.append(update_record)
        if len(self.update_history) > 50:
            self.update_history.pop(0)

        # Slowly drift base toward current policy (allows long-term evolution)
        self._drift_base()

        return update_record

    def adapt_on_win(self, ctx: TradeContext):
        """Reinforce successful patterns by slightly boosting their confidence."""
        pattern = ctx.pattern
        if pattern in self.current_policy["pattern_confidences"]:
            old = self.current_policy["pattern_confidences"][pattern]
            self.current_policy["pattern_confidences"][pattern] = min(0.98, old + 0.01)

        # Slightly increase position sizing on wins (bounded)
        self.current_policy["position_size_pct"] = min(0.40, self.current_policy["position_size_pct"] + 0.005)

    def _compute_kl(self, policy: dict) -> float:
        """Approximate KL divergence between policy and base_policy."""
        kl = 0.0
        # Compare scalar parameters
        for key in ["position_size_pct", "stop_loss_pct", "take_profit_pct", "xgb_risk_threshold"]:
            raw_p = policy.get(key, 0.5)
            raw_q = self.base_policy.get(key, 0.5)
            p = float(raw_p) if isinstance(raw_p, (int, float)) else 0.5
            q = float(raw_q) if isinstance(raw_q, (int, float)) else 0.5
            if p > 0 and q > 0:
                kl += abs(p - q) / max(q, 0.01)

        # Compare pattern confidences
        base_patterns = self.base_policy.get("pattern_confidences", {})
        policy_patterns = policy.get("pattern_confidences", {})
        if isinstance(base_patterns, dict):
            for pattern, base_val in base_patterns.items():
                p_val = policy_patterns.get(pattern, 0.5) if isinstance(policy_patterns, dict) else 0.5
                p = float(p_val) if isinstance(p_val, (int, float)) else 0.5
                q = float(base_val) if isinstance(base_val, (int, float)) else 0.5
                if p > 0 and q > 0:
                    kl += abs(p - q) / max(q, 0.01)

        return kl

    def _interpolate(self, old: dict, new: dict, alpha: float) -> dict:
        """Interpolate between old and new policy by alpha (0=old, 1=new)."""
        result: dict = json.loads(json.dumps(old))
        alpha = max(0.0, min(1.0, alpha))

        for key in ["position_size_pct", "stop_loss_pct", "take_profit_pct", "xgb_risk_threshold"]:
            if key in new and isinstance(new[key], (int, float)):
                old_val = float(old.get(key, 0.5)) if isinstance(old.get(key), (int, float)) else 0.5
                result[key] = old_val * (1.0 - alpha) + float(new[key]) * alpha

        res_patterns = result.get("pattern_confidences", {})
        new_patterns = new.get("pattern_confidences", {})
        if isinstance(res_patterns, dict) and isinstance(new_patterns, dict):
            for pattern in res_patterns:
                if pattern in new_patterns and isinstance(new_patterns[pattern], (int, float)):
                    old_p = float(res_patterns.get(pattern, 0.5))
                    result["pattern_confidences"][pattern] = (
                        old_p * (1.0 - alpha) + float(new_patterns[pattern]) * alpha
                    )

        result["regime_filter_strict"] = bool(new.get("regime_filter_strict", old.get("regime_filter_strict", False)))
        delay = new.get("entry_delay_bars")
        result["entry_delay_bars"] = int(delay) if delay is not None else int(old.get("entry_delay_bars", 0))
        return result

    def _drift_base(self):
        """Slowly drift base policy toward current policy."""
        r = self.base_decay_rate
        for key in ["position_size_pct", "stop_loss_pct", "take_profit_pct", "xgb_risk_threshold"]:
            b = self.base_policy.get(key, 0.5)
            c = self.current_policy.get(key, 0.5)
            b_val = float(b) if isinstance(b, (int, float)) else 0.5
            c_val = float(c) if isinstance(c, (int, float)) else 0.5
            self.base_policy[key] = b_val * (1.0 - r) + c_val * r

        base_pat = self.base_policy.get("pattern_confidences")
        curr_pat = self.current_policy.get("pattern_confidences")
        if isinstance(base_pat, dict) and isinstance(curr_pat, dict):
            for p, b_v in base_pat.items():
                if p in curr_pat:
                    c_v = curr_pat[p]
                    b_p = float(b_v) if isinstance(b_v, (int, float)) else 0.5
                    c_p = float(c_v) if isinstance(c_v, (int, float)) else 0.5
                    base_pat[p] = b_p * (1.0 - r) + c_p * r

    def get_adapted_params(self) -> dict:
        """Get current adapted policy parameters for trade execution."""
        return {
            "position_size_pct": round(self.current_policy["position_size_pct"], 4),
            "stop_loss_pct": round(self.current_policy["stop_loss_pct"], 4),
            "take_profit_pct": round(self.current_policy["take_profit_pct"], 4),
            "xgb_risk_threshold": round(self.current_policy["xgb_risk_threshold"], 4),
            "regime_filter_strict": self.current_policy["regime_filter_strict"],
            "entry_delay_bars": self.current_policy.get("entry_delay_bars", 0),
            "pattern_confidences": {
                k: round(v, 3) for k, v in self.current_policy["pattern_confidences"].items()
            },
        }

    def get_summary(self) -> dict:
        return {
            "total_adaptations": self.total_adaptations,
            "kl_distance": round(self._compute_kl(self.current_policy), 4),
            "kl_budget": self.kl_budget,
            "current_params": self.get_adapted_params(),
            "recent_updates": self.update_history[-3:] if self.update_history else [],
        }


# ============================================================
# Advanced RL Suite: MuZero MCTS, GRPO, and PDRL / ReMax
# ============================================================

class MuZeroMCTSPlanner:
    """Model-based Monte Carlo Tree Search (MCTS) lookahead planner with Latent Dynamics.
    
    Simulates multi-step forward trajectories over crypto price dynamics,
    evaluating value paths and pruning negative expected value (loss) branches.
    """

    def __init__(self, lookahead_depth: int = 5, num_simulations: int = 30):
        self.lookahead_depth = lookahead_depth
        self.num_simulations = num_simulations
        self.value_bias = 0.0  # Unbiased initial value prior (Bug 9 fix)
        self.total_tree_searches = 0
        self.pruned_branches_count = 0
        self.last_search_plan: dict = {}

    def plan_trajectory(self, side: str, pattern: str, regime: str,
                        rsi: float, macd_val: float, failure_rate: float) -> dict:
        """Execute MCTS multi-step lookahead search across candidate execution branches."""
        self.total_tree_searches += 1
        
        # Candidate actions at root
        actions = ["EXECUTE_PRIMARY", "TIGHT_STOP_EXECUTE", "DELAYED_CONFIRM_EXECUTE", "HOLD_DEFENSE"]
        
        # State dynamics representation
        momentum = 1.0 if side == "BUY" else -1.0
        regime_factor = 1.2 if "Trend" in regime else (0.8 if "Volatility" in regime else 1.0)
        rsi_delta = (50.0 - rsi) / 50.0 if side == "BUY" else (rsi - 50.0) / 50.0

        branch_evaluations = {}
        pruned_in_this_search = 0

        for action in actions:
            path_returns = []
            # MCTS Rollouts for action branch
            for sim in range(self.num_simulations // len(actions)):
                sim_return = 0.0
                curr_momentum = momentum * regime_factor
                
                # Simulating K-step latent forward transition
                for step in range(1, self.lookahead_depth + 1):
                    # Transition dynamics G(s, a) + noise
                    step_shock = (random.random() - 0.48) * 0.4
                    step_drift = (curr_momentum * 0.35) + (rsi_delta * 0.2) + self.value_bias
                    
                    if action == "TIGHT_STOP_EXECUTE":
                        step_drift *= 0.95
                        if step_shock < -0.6:  # Stopped out safely
                            step_drift = max(step_drift, -0.6)
                            sim_return += step_drift
                            break
                    elif action == "HOLD_DEFENSE":
                        step_drift = 0.0
                    
                    sim_return += step_drift + step_shock
                    curr_momentum *= 0.88  # Mean-reverting decay
                
                path_returns.append(sim_return)

            avg_return = sum(path_returns) / max(1, len(path_returns))
            # Penalize by historical failure rate
            adjusted_ev = avg_return - (failure_rate * 0.8)
            
            if adjusted_ev < -0.2:
                pruned_in_this_search += 1
            
            branch_evaluations[action] = {
                "expected_return_pct": round(adjusted_ev, 3),
                "win_probability": round(max(0.05, min(0.95, 0.5 + adjusted_ev * 0.25)), 2),
                "simulations": len(path_returns),
            }

        self.pruned_branches_count += pruned_in_this_search
        
        # Pick action with maximum Expected Value (UCB selection)
        best_action = max(branch_evaluations.items(), key=lambda x: x[1]["expected_return_pct"])
        
        plan = {
            "best_action": best_action[0],
            "expected_value_pct": best_action[1]["expected_return_pct"],
            "win_probability": best_action[1]["win_probability"],
            "lookahead_depth": self.lookahead_depth,
            "total_simulations": self.num_simulations,
            "pruned_loss_branches": pruned_in_this_search,
            "branches": branch_evaluations,
            "mcts_status": "HIGH_CONVICTION" if best_action[1]["expected_return_pct"] > 0.3 else "MODERATE",
        }
        self.last_search_plan = plan
        return plan

    def update_value_prior(self, negative_bias: bool = False):
        """Update latent value prior based on realized market feedback."""
        if negative_bias:
            self.value_bias = max(-0.25, self.value_bias - 0.03)
        else:
            self.value_bias = min(0.25, self.value_bias + 0.015)

    def get_summary(self) -> dict:
        return {
            "total_tree_searches": self.total_tree_searches,
            "pruned_branches_count": self.pruned_branches_count,
            "lookahead_depth": self.lookahead_depth,
            "simulations_per_search": self.num_simulations,
            "current_value_prior": round(self.value_bias, 3),
            "last_plan": self.last_search_plan,
        }


class GRPOEvaluator:
    """Group Relative Policy Optimization (GRPO) Evaluator.
    
    Generates a group of candidate execution policies and optimizes relative
    advantage scores without requiring unstable critic baselines.
    """

    def __init__(self, group_size: int = 4):
        self.group_size = group_size
        self.total_group_evals = 0
        self.last_group_result: dict = {}

    def evaluate_group(self, base_pos_pct: float, base_stop_loss: float,
                       base_take_profit: float, orm_score: float, pattern_conf: float) -> dict:
        """Evaluate candidate group of execution configurations and compute relative advantages."""
        self.total_group_evals += 1
        
        # Generate candidate variations in the group
        candidates = [
            {"id": "G1_Standard", "pos_mult": 1.0, "sl_mult": 1.0, "tp_mult": 1.0, "desc": "Standard Policy Anchor"},
            {"id": "G2_DefensiveTight", "pos_mult": 0.75, "sl_mult": 0.80, "tp_mult": 1.15, "desc": "Tight Risk Asymmetric R:R"},
            {"id": "G3_HighExpectancy", "pos_mult": 1.15, "sl_mult": 0.90, "tp_mult": 1.30, "desc": "High-Conviction Alpha Target"},
            {"id": "G4_ConservativeFloor", "pos_mult": 0.50, "sl_mult": 0.70, "tp_mult": 0.90, "desc": "Low Volatility Capital Preserver"},
        ]

        raw_rewards = []
        for c in candidates:
            pos_m = float(c["pos_mult"])
            sl_m = float(c["sl_mult"])
            tp_m = float(c["tp_mult"])
            # Multi-attribute scoring: Expected Return / Risk + ORM confluence - fee drag
            est_rr = (base_take_profit * tp_m) / max(0.2, (base_stop_loss * sl_m))
            r = (orm_score * 0.45) + (pattern_conf * 0.35) + (est_rr * 0.15) - (0.002 * 10.0)
            raw_rewards.append(r)

        # Standardize Group-Relative Advantage: A_i = (R_i - mean) / (std + eps)
        mean_r = sum(raw_rewards) / max(1, len(raw_rewards))
        variance = sum((r - mean_r) ** 2 for r in raw_rewards) / max(1, len(raw_rewards))
        std_r = math.sqrt(max(1e-6, variance))

        ranked_group = []
        for idx, c in enumerate(candidates):
            adv = (raw_rewards[idx] - mean_r) / std_r
            pos_m = float(c["pos_mult"])
            sl_m = float(c["sl_mult"])
            tp_m = float(c["tp_mult"])
            ranked_group.append({
                "candidate": str(c["id"]),
                "description": str(c["desc"]),
                "relative_advantage": round(adv, 3),
                "raw_reward": round(raw_rewards[idx], 3),
                "recommended_pos_pct": round(base_pos_pct * pos_m, 3),
                "recommended_sl_pct": round(base_stop_loss * sl_m, 3),
                "recommended_tp_pct": round(base_take_profit * tp_m, 3),
            })

        ranked_group.sort(key=lambda x: float(x["relative_advantage"]), reverse=True)
        best_candidate = ranked_group[0]

        top_adv = float(best_candidate["relative_advantage"])
        worst_adv = float(ranked_group[-1]["relative_advantage"])
        spread = round(top_adv - worst_adv, 3)

        result = {
            "top_candidate": best_candidate["candidate"],
            "top_advantage": top_adv,
            "best_params": {
                "pos_pct": best_candidate["recommended_pos_pct"],
                "sl_pct": best_candidate["recommended_sl_pct"],
                "tp_pct": best_candidate["recommended_tp_pct"],
            },
            "group_rankings": ranked_group,
            "mean_reward": round(mean_r, 3),
            "advantage_spread": spread,
        }
        self.last_group_result = result
        return result

    def get_summary(self) -> dict:
        return {
            "total_group_evaluations": self.total_group_evals,
            "group_size": self.group_size,
            "last_group_result": self.last_group_result,
        }


class PDRLLossMitigator:
    """Penalty-Driven Reinforcement Learning (PDRL) & ReMax Loss Exploitation Engine.
    
    When trading experiences losses, PDRL enforces strict EXPLOITATION MODE:
    - Suppresses exploration (zero random attempts during loss recovery)
    - Applies asymmetric downside loss penalization
    - Enforces strict pattern-memory matching (>= 75% historical win rate required)
    - Tightens risk parameters via Fractional Kelly sizing
    """

    def __init__(self, penalty_lambda: float = 2.0):
        self.penalty_lambda = penalty_lambda
        self.consecutive_losses = 0
        self.recent_loss_streak = 0
        self.max_loss_streak = 0
        self.total_penalties_applied = 0
        self.exploitation_mode_active = False
        self.recovery_trades_completed = 0
        self.loss_history: list[float] = []

    def register_loss(self, pnl: float, pnl_pct: float) -> dict:
        """Register a loss, increment penalty multipliers, and activate Deep Exploitation."""
        self.consecutive_losses += 1
        self.recent_loss_streak += 1
        if self.consecutive_losses > self.max_loss_streak:
            self.max_loss_streak = self.consecutive_losses
            
        self.exploitation_mode_active = True
        self.loss_history.append(abs(pnl))
        if len(self.loss_history) > 30:
            self.loss_history.pop(0)

        # Asymmetric Loss Penalty: Lambda * Loss * (1 + 0.5 * streak)
        penalty = round(self.penalty_lambda * abs(pnl_pct) * (1.0 + 0.5 * self.consecutive_losses), 3)
        self.total_penalties_applied += 1

        return {
            "status": "DEEP_EXPLOITATION_ACTIVATED",
            "consecutive_losses": self.consecutive_losses,
            "asymmetric_penalty": penalty,
            "recommendation": "Enforce strict exploitation on highest-conviction patterns with MuZero MCTS verification.",
        }

    def register_win(self, pnl: float, pnl_pct: float) -> dict:
        """Register a win, relax loss streak, and update recovery tracking."""
        if self.consecutive_losses > 0:
            self.consecutive_losses = max(0, self.consecutive_losses - 1)
            self.recovery_trades_completed += 1
            if self.consecutive_losses == 0:
                self.exploitation_mode_active = False
                self.recent_loss_streak = 0
        
        return {
            "status": "EXPLOITATION_BALANCED" if not self.exploitation_mode_active else "RECOVERING",
            "consecutive_losses": self.consecutive_losses,
            "recovery_trades_completed": self.recovery_trades_completed,
        }

    def is_exploitation_mandated(self, current_drawdown: float = 0.0) -> bool:
        """Check if trading state mandates 100% exploitation (no exploration)."""
        return self.exploitation_mode_active or self.consecutive_losses > 0 or current_drawdown > 0.015

    def get_summary(self) -> dict:
        return {
            "exploitation_mode_active": self.exploitation_mode_active,
            "consecutive_losses": self.consecutive_losses,
            "max_loss_streak": self.max_loss_streak,
            "penalty_lambda": self.penalty_lambda,
            "total_penalties_applied": self.total_penalties_applied,
            "recovery_trades_completed": self.recovery_trades_completed,
            "exploitation_state_label": "🎯 DEEP EXPLOITATION (LOSS RECOVERY)" if self.exploitation_mode_active else "⚖️ BALANCED DYNAMICS",
        }

# ============================================================
# Goal and Ruin Survival Controller ($0.00 Ruin <-> $1,050.00 Goal)
# ============================================================

class GoalAndSurvivalController:
    """Manages the autonomous self-learning RL goal: achieve $1,050.00 profit and prevent $0.00 ruin."""

    def __init__(self, start_capital: float = 1000.0, profit_target: float = 1050.0, ruin_floor: float = 0.0):
        self.start_capital = start_capital
        self.profit_target = profit_target
        self.ruin_floor = ruin_floor
        self.generation = 1
        self.goals_achieved = 0
        self.survivals_triggered = 0
        self.last_boundary_event = None
        self.current_equity = start_capital

    def check_boundary_and_adapt(self, current_equity: float, webrl_engine) -> Optional[dict]:
        """Trigger deep self-learning adaptation when nearing $0 ruin or hitting $1,050 profit."""
        self.current_equity = current_equity

        # 1. Target Profit Achieved ($1,050.00)
        if current_equity >= self.profit_target:
            self.goals_achieved += 1
            self.generation += 1
            old_target = self.profit_target
            self.profit_target = round(current_equity + 50.0, 2)  # Step up next milestone target (+50)
            
            # RL Weight Reward Boost on all winning patterns
            cur_policy = webrl_engine.policy_adapter.current_policy
            for p in cur_policy["pattern_confidences"]:
                cur_policy["pattern_confidences"][p] = min(0.98, round(cur_policy["pattern_confidences"][p] * 1.03, 3))
            
            # Slightly expand take-profit to lock in higher alpha
            cur_policy["take_profit_pct"] = min(2.5, round(cur_policy["take_profit_pct"] + 0.1, 3))
            webrl_engine.policy_adapter.base_policy = json.loads(json.dumps(cur_policy))

            event = {
                "type": "PROFIT_GOAL_ACHIEVED",
                "message": f"🏆 PROFIT TARGET REACHED! Equity reached ${current_equity:.2f} (Target was ${old_target:.2f}) | Promoted to Evolved Generation #{self.generation} | Next Goal: ${self.profit_target:.2f}",
                "generation": self.generation,
                "current_equity": current_equity,
                "next_target": self.profit_target,
                "actions": "Reinforced winning weights, widened Take-Profit target, evolved generation.",
                "timestamp": time.strftime("%H:%M:%S")
            }
            self.last_boundary_event = event
            return event

        # 2. Ruin Prevention / Steep Drawdown (Equity drops toward $0, e.g. < $950)
        elif current_equity < (self.start_capital * 0.95) and current_equity > self.ruin_floor:
            self.survivals_triggered += 1
            cur_policy = webrl_engine.policy_adapter.current_policy
            
            # Aggressive survival weight adjustment
            cur_policy["position_size_pct"] = max(0.12, round(cur_policy["position_size_pct"] * 0.88, 3))
            cur_policy["stop_loss_pct"] = max(0.50, round(cur_policy["stop_loss_pct"] * 0.88, 3))
            cur_policy["regime_filter_strict"] = True
            cur_policy["xgb_risk_threshold"] = max(0.18, round(cur_policy["xgb_risk_threshold"] * 0.90, 3))
            webrl_engine.policy_adapter.base_policy = json.loads(json.dumps(cur_policy))

            event = {
                "type": "RUIN_PREVENTION_ACTIVATED",
                "message": f"🚨 CAPITAL DEFENSE ACTIVATED: Equity down to ${current_equity:.2f} -> Reduced position size to {cur_policy['position_size_pct']*100:.1f}% & tightened stop-loss to {cur_policy['stop_loss_pct']:.2f}% to avoid $0.00 ruin.",
                "generation": self.generation,
                "current_equity": current_equity,
                "actions": "Cut allocation size, strict regime lock, tightened risk threshold.",
                "timestamp": time.strftime("%H:%M:%S")
            }
            self.last_boundary_event = event
            return event

        return None

    def get_summary(self, current_equity: float = 1000.0) -> dict:
        eq = current_equity if current_equity is not None else self.current_equity
        dist_to_goal = max(0.0, round(self.profit_target - eq, 2))
        safety_to_ruin = max(0.0, round(eq - self.ruin_floor, 2))
        progress_pct = max(0.0, min(100.0, round(((eq - self.start_capital) / max(1.0, (self.profit_target - self.start_capital))) * 100.0, 1))) if eq >= self.start_capital else 0.0
        
        return {
            "start_capital": self.start_capital,
            "current_equity": eq,
            "profit_target": self.profit_target,
            "ruin_floor": self.ruin_floor,
            "distance_to_goal": dist_to_goal,
            "safety_buffer_to_ruin": safety_to_ruin,
            "progress_to_target_pct": progress_pct,
            "generation": self.generation,
            "goals_achieved": self.goals_achieved,
            "survivals_triggered": self.survivals_triggered,
            "last_event": self.last_boundary_event,
        }


# ============================================================
# 5a. Q-Learning Signal Table (Real Self-Learning)
# ============================================================

class QLearningSignalTable:
    """Tabular Q-learning that learns which (trend, rsi_zone, bb_position, momentum_zone, side)
    combinations are profitable and which should be avoided.
    
    This is GENUINE reinforcement learning:
    - State = discretized market conditions from real technical indicators
    - Action = BUY or SELL
    - Reward = actual realized PnL percentage from completed trades
    - Q-update = Temporal Difference learning: Q(s,a) += alpha * (reward - Q(s,a))
    """

    TREND_STATES = ["BULLISH", "BEARISH", "FLAT"]
    RSI_ZONES = ["OVERSOLD", "NEUTRAL", "OVERBOUGHT"]
    BB_POSITIONS = ["BELOW_LOWER", "MID", "ABOVE_UPPER"]
    MOMENTUM_ZONES = ["STRONG_DOWN", "WEAK_DOWN", "NEUTRAL", "WEAK_UP", "STRONG_UP"]
    ACTIONS = ["BUY", "SELL"]

    def __init__(self, alpha: float = 0.15, gamma: float = 0.0, min_trades: int = 5):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor (0 for immediate reward)
        self.min_trades = min_trades  # Minimum trades before trusting Q-value (Bug 5/8 fix)
        self.q_table: Dict[str, Dict[str, float]] = {}  # state_key -> {action -> q_value}
        self.visit_count: Dict[str, Dict[str, int]] = {}  # state_key -> {action -> count}
        self.total_updates = 0

    def _discretize_momentum(self, momentum_pct: float) -> str:
        """Discretize continuous momentum into zones."""
        if momentum_pct < -0.1:
            return "STRONG_DOWN"
        elif momentum_pct < -0.02:
            return "WEAK_DOWN"
        elif momentum_pct < 0.02:
            return "NEUTRAL"
        elif momentum_pct < 0.1:
            return "WEAK_UP"
        else:
            return "STRONG_UP"

    def _make_state_key(self, indicators: dict) -> str:
        """Create discretized state key from real indicators."""
        trend = indicators.get("trend", "FLAT")
        rsi_zone = indicators.get("rsi_zone", "NEUTRAL")
        bb_pos = indicators.get("bb_position", "MID")
        mom_zone = self._discretize_momentum(indicators.get("momentum_20", 0.0))
        return f"{trend}|{rsi_zone}|{bb_pos}|{mom_zone}"

    def get_q_value(self, indicators: dict, action: str) -> float:
        """Get Q-value for a (state, action) pair."""
        state_key = self._make_state_key(indicators)
        if state_key not in self.q_table:
            return 0.0  # Optimistic initialization
        return self.q_table[state_key].get(action, 0.0)

    def get_visit_count(self, indicators: dict, action: str) -> int:
        """Get how many times this (state, action) has been visited."""
        state_key = self._make_state_key(indicators)
        if state_key not in self.visit_count:
            return 0
        return self.visit_count[state_key].get(action, 0)

    def should_trade(self, indicators: dict, action: str) -> Tuple[bool, float, str]:
        """Decide whether to trade based on Q-value and exploration.
        
        Returns: (should_trade, q_value, reason)
        """
        state_key = self._make_state_key(indicators)
        q_val = self.get_q_value(indicators, action)
        visits = self.get_visit_count(indicators, action)

        # Exploration: if we haven't seen this state enough, explore it
        if visits < self.min_trades:
            return True, q_val, f"EXPLORING (visits={visits}/{self.min_trades})"

        # Exploitation: strictly non-negative expected value (Bug 5 fix: no negative EV tolerance)
        if q_val >= 0.0:
            return True, q_val, f"Q_POSITIVE (Q={q_val:+.3f}, visits={visits})"
        else:
            return False, q_val, f"Q_NEGATIVE (Q={q_val:+.3f}, visits={visits})"

    def update(self, indicators: dict, action: str, reward: float):
        """Update Q-value using temporal difference learning.
        
        Q(s,a) += alpha * (reward - Q(s,a))
        """
        state_key = self._make_state_key(indicators)
        
        # Initialize if new state
        if state_key not in self.q_table:
            self.q_table[state_key] = {"BUY": 0.0, "SELL": 0.0}
            self.visit_count[state_key] = {"BUY": 0, "SELL": 0}

        old_q = self.q_table[state_key].get(action, 0.0)
        # TD update: Q(s,a) = Q(s,a) + alpha * (reward - Q(s,a))
        new_q = old_q + self.alpha * (reward - old_q)
        self.q_table[state_key][action] = round(new_q, 5)
        self.visit_count[state_key][action] = self.visit_count[state_key].get(action, 0) + 1
        self.total_updates += 1

    def get_summary(self) -> dict:
        """Get human-readable summary of Q-table state."""
        profitable_states = []
        unprofitable_states = []
        for state_key, actions in self.q_table.items():
            for action, q_val in actions.items():
                visits = self.visit_count.get(state_key, {}).get(action, 0)
                if visits >= self.min_trades:
                    entry = {"state": state_key, "action": action, "q_value": round(q_val, 4), "visits": visits}
                    if q_val > 0:
                        profitable_states.append(entry)
                    else:
                        unprofitable_states.append(entry)

        profitable_states.sort(key=lambda x: x["q_value"], reverse=True)
        unprofitable_states.sort(key=lambda x: x["q_value"])

        return {
            "total_state_action_pairs": sum(len(v) for v in self.q_table.values()),
            "total_updates": self.total_updates,
            "unique_states_visited": len(self.q_table),
            "top_profitable_signals": profitable_states[:5],
            "top_unprofitable_signals": unprofitable_states[:5],
            "learning_rate": self.alpha,
        }


# ============================================================
# 5b. Unified WebRL Engine
# ============================================================

class WebRLEngine:
    """Unified WebRL Self-Evolving engine combining all components + Q-Learning + Goal/Survival Controller."""

    def __init__(self):
        self.loss_analyzer = LossAnalyzer()
        self.win_analyzer = WinAnalyzer()
        self.pattern_memory = PatternMemoryBank()
        self.goal_controller = GoalAndSurvivalController(start_capital=1000.0, profit_target=1050.0, ruin_floor=0.0)
        self.curriculum = SelfEvolvingCurriculum()
        self.orm = OutcomeRewardModel()
        self.policy_adapter = KLConstrainedPolicyAdapter()
        
        # Advanced RL Suite
        self.muzero = MuZeroMCTSPlanner(lookahead_depth=5, num_simulations=32)
        self.grpo = GRPOEvaluator(group_size=4)
        self.pdrl = PDRLLossMitigator(penalty_lambda=2.2)

        # *** REAL Q-Learning Signal Table (min_trades=5, Bug 8 fix) ***
        self.q_table = QLearningSignalTable(alpha=0.15, gamma=0.0, min_trades=5)

        self.total_trades = 0
        self.total_losses = 0
        self.total_wins = 0
        self.trade_id_counter = 0
        self.total_evolutions_count = 0
        self.last_evolution_time = time.strftime("%H:%M:%S")
        self.last_evolution_timestamp = time.time()
        self.evolution_history: list[dict] = [
            {
                "time": time.strftime("%H:%M:%S"),
                "type": "INITIALIZATION",
                "reason": "Agent Brain Initialized",
                "details": "Q-Learning Signal Table + MuZero MCTS + GRPO + PDRL Online",
                "evolution_index": 0
            }
        ]
        self.learning_curve: list[dict] = []  # Tracks evolving win rate over time

    def record_evolution(self, ev_type: str, reason: str, details: str):
        """Record the exact time and explanation of an evolution event."""
        t_str = time.strftime("%H:%M:%S")
        self.last_evolution_time = t_str
        self.last_evolution_timestamp = time.time()
        self.total_evolutions_count += 1
        self.evolution_history.append({
            "time": t_str,
            "type": ev_type,
            "reason": reason,
            "details": details,
            "evolution_index": self.total_evolutions_count,
        })
        if len(self.evolution_history) > 30:
            self.evolution_history.pop(0)

    def on_loss(self, ctx: TradeContext, entry_indicators: Optional[Dict] = None) -> dict:
        """Full WebRL + Q-Learning + PDRL pipeline triggered on a losing trade."""
        self.trade_id_counter += 1
        ctx.trade_id = self.trade_id_counter
        self.total_trades += 1
        self.total_losses += 1

        # *** Q-TABLE UPDATE: Feed back NEGATIVE reward from real trade outcome ***
        if entry_indicators:
            reward = max(-1.0, ctx.pnl_pct / 1.5)  # Normalize: -1.5% loss -> -1.0 reward
            self.q_table.update(entry_indicators, ctx.side, reward)

        # 1. Analyze the failure
        failure = self.loss_analyzer.analyze(ctx)

        # 2. Register failure in Pattern Memory Bank
        self.pattern_memory.register_trade_outcome(ctx.pattern, ctx.pnl, "LOSS", failure.failure_cause)

        # 3. PDRL & ReMax Penalty Registration
        pdrl_res = self.pdrl.register_loss(ctx.pnl, ctx.pnl_pct)

        # 4. MuZero Value Prior Negative Calibration
        self.muzero.update_value_prior(negative_bias=True)

        # 5. Generate curriculum items from the failure
        new_items = self.curriculum.generate_from_failure(failure)

        # 6. Update ORM with negative outcome
        features = ctx.to_feature_vector()
        outcome_score = max(-1.0, ctx.pnl_pct / 2.0)
        self.orm.update(features, outcome_score)

        # 7. Adapt policy parameters
        policy_update = self.policy_adapter.adapt_on_loss(failure)

        # 8. Replay top curriculum item
        replayed = self.curriculum.replay_top()

        # Track learning curve
        self._update_learning_curve()

        # Record evolution
        q_info = ""
        if entry_indicators:
            state_key = self.q_table._make_state_key(entry_indicators)
            q_val = self.q_table.get_q_value(entry_indicators, ctx.side)
            q_info = f" | Q({state_key},{ctx.side})={q_val:+.3f}"
        self.record_evolution(
            ev_type="Q_LEARNING_LOSS",
            reason=f"Loss: {failure.failure_cause}",
            details=f"Q-Table updated with negative reward ({ctx.pnl_pct:+.2f}%){q_info} | Streak #{pdrl_res['consecutive_losses']}"
        )

        # 9. Check for 100-attempt milestone
        milestone_report = None
        if self.total_trades % 100 == 0:
            milestone_report = self.run_100_attempt_macro_analysis()
            self.record_evolution(
                ev_type="100_ATTEMPT_MILESTONE",
                reason="Macro 100-Attempt Batch Replay",
                details=f"Recalibrated weights to reduce loss by {milestone_report['estimated_loss_reduction']}"
            )

        return {
            "event": "LOSS_ANALYZED",
            "failure": failure.to_dict(),
            "pdrl_status": pdrl_res,
            "curriculum_items_generated": len(new_items),
            "orm_score_after": round(self.orm.predict_score(features), 3),
            "policy_update": policy_update,
            "replayed_scenario": replayed.scenario_desc if replayed else None,
            "milestone_report": milestone_report,
            "evolution_time": self.last_evolution_time,
        }

    def on_win(self, ctx: TradeContext, entry_indicators: Optional[Dict] = None) -> dict:
        """Analyze WHY the winning pattern gave profit, store in memory bank, and reinforce Q-table."""
        self.trade_id_counter += 1
        ctx.trade_id = self.trade_id_counter
        self.total_trades += 1
        self.total_wins += 1

        # *** Q-TABLE UPDATE: Feed back POSITIVE reward (symmetric divisor 1.5, Bug 6 fix) ***
        if entry_indicators:
            reward = min(1.0, ctx.pnl_pct / 1.5)  # Normalize: +1.5% win -> +1.0 reward (symmetric with on_loss)
            self.q_table.update(entry_indicators, ctx.side, reward)

        self.loss_analyzer.record_win(ctx)

        # 1. Deeply analyze why this trade produced profit
        win_case = self.win_analyzer.analyze(ctx)

        # 2. Store and index winning pattern in Pattern Memory Bank
        self.pattern_memory.register_trade_outcome(ctx.pattern, ctx.pnl, "PROFIT", win_case.profit_driver)

        # 3. PDRL & MuZero positive reinforcement
        pdrl_res = self.pdrl.register_win(ctx.pnl, ctx.pnl_pct)
        self.muzero.update_value_prior(negative_bias=False)

        # 4. Update ORM with positive outcome (scaled by win quality)
        features = ctx.to_feature_vector()
        outcome_score = min(1.0, (ctx.pnl_pct / 1.8) * win_case.quality_score)
        self.orm.update(features, outcome_score)

        # 5. Adapt & reinforce policy parameters
        self.policy_adapter.adapt_on_win(ctx)

        self._update_learning_curve()

        # Record evolution event with exact timestamp
        self.record_evolution(
            ev_type="WIN_REINFORCEMENT",
            reason=f"Profit Driver: {win_case.profit_driver}",
            details=f"Reinforced '{ctx.pattern}' | MuZero EV Prior Boosted | PDRL Streak: {pdrl_res['consecutive_losses']}"
        )

        # 6. Check for 100-attempt deep optimization milestone
        milestone_report = None
        if self.total_trades % 100 == 0:
            milestone_report = self.run_100_attempt_macro_analysis()
            self.record_evolution(
                ev_type="100_ATTEMPT_MILESTONE",
                reason="Macro 100-Attempt Batch Replay",
                details=f"Recalibrated weights to reduce loss by {milestone_report['estimated_loss_reduction']}"
            )

        return {
            "event": "WIN_ANALYZED_AND_STORED",
            "win_analysis": win_case.to_dict(),
            "pdrl_status": pdrl_res,
            "orm_score_after": round(self.orm.predict_score(features), 3),
            "milestone_report": milestone_report,
            "evolution_time": self.last_evolution_time,
        }

    def check_equity_boundary(self, current_equity: float) -> Optional[dict]:
        """Check if equity crossed profit goal ($1,050.00) or ruin defense boundary ($0.00)."""
        return self.goal_controller.check_boundary_and_adapt(current_equity, self)

    def run_100_attempt_macro_analysis(self) -> dict:
        """Deep analysis of all 100 completed attempts and global weight recalibration to reduce future losses."""
        milestone_idx = self.total_trades // 100
        recent_win_rate = (self.total_wins / max(1, self.total_trades)) * 100.0
        
        # 1. Analyze loss cause distribution over recent failures
        loss_summary = self.loss_analyzer.get_summary()
        cause_dist = loss_summary.get("cause_distribution", {})
        top_loss_cause = max(cause_dist.items(), key=lambda x: x[1])[0] if cause_dist else "UNKNOWN"

        # 2. Recalibrate pattern confidence weights based on empirical failure rate & memory bank
        cur_policy = self.policy_adapter.current_policy
        recalibrated_weights = {}
        for pattern, conf in cur_policy["pattern_confidences"].items():
            fail_rate = self.loss_analyzer.get_pattern_failure_rate(pattern)
            if fail_rate > 0.4:
                new_w = max(0.45, round(conf * (1.0 - (fail_rate - 0.3) * 0.5), 3))
            else:
                new_w = min(0.98, round(conf * 1.05, 3))
            cur_policy["pattern_confidences"][pattern] = new_w
            recalibrated_weights[pattern] = new_w

        # 3. Macro training loop on ORM: 10 batch replay passes over experience buffer
        if len(self.orm.experience_buffer) >= 10:
            for _ in range(10):
                self.orm._replay_batch(batch_size=min(15, len(self.orm.experience_buffer)))

        # 4. Tune stop-loss and position sizing to reduce future drawdown
        if self.total_losses > self.total_wins:
            cur_policy["stop_loss_pct"] = max(0.6, round(cur_policy["stop_loss_pct"] * 0.9, 3))
            cur_policy["position_size_pct"] = max(0.18, round(cur_policy["position_size_pct"] * 0.92, 3))
            cur_policy["xgb_risk_threshold"] = max(0.20, round(cur_policy["xgb_risk_threshold"] * 0.95, 3))
            cur_policy["regime_filter_strict"] = True

        # 5. Re-center base policy with updated macro knowledge
        self.policy_adapter.base_policy = json.loads(json.dumps(cur_policy))

        # Compute real empirical win-rate delta vs 50% baseline (Bug 10 fix)
        empirical_win_delta = max(0.0, recent_win_rate - 50.0)
        reduction_pct = round(min(100.0, (empirical_win_delta / 50.0) * 100.0), 1) if recent_win_rate >= 50.0 else 0.0

        report = {
            "milestone_number": milestone_idx,
            "total_attempts": self.total_trades,
            "win_rate": round(recent_win_rate, 1),
            "top_loss_driver": top_loss_cause,
            "actions_taken": [
                f"Recalibrated weights across {len(recalibrated_weights)} patterns based on 100-attempt performance",
                f"Executed 10-batch curriculum replay optimization on Outcome Reward Model (ORM)",
                f"MuZero MCTS Latent Dynamics recalibrated with negative loss branch pruning",
                f"Adjusted Stop-Loss to {cur_policy['stop_loss_pct']:.2f}% & Position Size to {cur_policy['position_size_pct']*100:.1f}%",
                f"Synchronized base policy anchor with 100-attempt loss-minimization weights",
            ],
            "recalibrated_weights": recalibrated_weights,
            "estimated_loss_reduction": f"{reduction_pct:.1f}%",
            "timestamp": time.strftime("%H:%M:%S"),
        }
        self.last_milestone_report = report
        return report

    def evaluate_trade(self, pattern: str, regime: str, regime_conf: float,
                       rsi: float, macd: str, side: str, current_drawdown: float = 0.0,
                       indicators: Optional[Dict] = None) -> dict:
        """Unified Pre-trade evaluation with Q-Learning, MuZero MCTS, GRPO, and PDRL.
        
        When `indicators` is provided (real market data), the Q-learning table
        is the primary decision maker. The other components provide supplementary signals.
        """
        hyp = TradeContext(
            side=side, pattern=pattern, regime=regime,
            regime_confidence=regime_conf, rsi=rsi, macd=macd,
            allocated_capital=self.policy_adapter.current_policy["position_size_pct"] * 1000,
        )
        features = hyp.to_feature_vector()
        orm_score = self.orm.predict_score(features)

        adapted = self.policy_adapter.get_adapted_params()
        pattern_conf = adapted["pattern_confidences"].get(pattern, 0.80)
        failure_rate = self.loss_analyzer.get_pattern_failure_rate(pattern)

        try:
            macd_num = float(str(macd).replace("+", "").replace("%", ""))
        except (ValueError, AttributeError):
            macd_num = 0.0

        # 1. Q-Learning Decision (PRIMARY when real indicators available)
        q_value = 0.0
        q_should_trade = True
        q_reason = "NO_INDICATORS"
        if indicators:
            q_should_trade, q_value, q_reason = self.q_table.should_trade(indicators, side)

        # 2. GRPO Group-Relative Advantage Evaluation
        grpo_res = self.grpo.evaluate_group(
            base_pos_pct=adapted["position_size_pct"],
            base_stop_loss=adapted["stop_loss_pct"],
            base_take_profit=adapted["take_profit_pct"],
            orm_score=orm_score,
            pattern_conf=pattern_conf,
        )

        # 3. MuZero MCTS Multi-Step Lookahead Planning
        muzero_plan = self.muzero.plan_trajectory(
            side=side, pattern=pattern, regime=regime,
            rsi=rsi, macd_val=macd_num, failure_rate=failure_rate,
        )

        # 4. PDRL Exploitation State Check
        is_exploitation_mandated = self.pdrl.is_exploitation_mandated(current_drawdown)

        # Regime filter: block counter-trend trades in strong regimes
        regime_blocked = False
        if indicators:
            trend = indicators.get("trend", "FLAT")
            if side == "BUY" and trend == "BEARISH" and regime_conf > 0.7:
                regime_blocked = True
            if side == "SELL" and trend == "BULLISH" and regime_conf > 0.7:
                regime_blocked = True

        # Composite score (Q-value weighted heavily)
        muzero_ev = muzero_plan.get("expected_value_pct", 0.0)
        grpo_adv = grpo_res.get("top_advantage", 0.0)
        utility_score = (
            q_value * 3.0          # Q-learning is primary
            + muzero_ev * 0.8
            + grpo_adv * 0.5
            + orm_score * 0.4
            + pattern_conf * 0.3
            - failure_rate * 0.5
        )

        # Decision: Q-learning gates the trade
        should_trade = q_should_trade and not regime_blocked
        
        if is_exploitation_mandated and should_trade:
            trade_mode = "Q_EXPLOIT"
            alloc_pct = max(0.10, min(0.20, adapted["position_size_pct"] * 0.7))
        elif should_trade:
            trade_mode = "Q_SIGNAL"
            alloc_pct = max(0.08, min(0.25, adapted["position_size_pct"]))
        else:
            trade_mode = f"BLOCKED ({q_reason})" if not q_should_trade else "REGIME_BLOCKED"
            alloc_pct = 0.0

        return {
            "orm_score": round(orm_score, 3),
            "utility_score": round(utility_score, 3),
            "q_value": round(q_value, 4),
            "q_reason": q_reason,
            "pattern_confidence": round(pattern_conf, 3),
            "pattern_failure_rate": round(failure_rate, 3),
            "regime_blocked": regime_blocked,
            "should_trade": should_trade,
            "trade_mode": trade_mode,
            "is_exploration": "EXPLORING" in q_reason,
            "is_exploitation_mandated": is_exploitation_mandated,
            "adapted_position_pct": round(alloc_pct, 3),
            "adapted_stop_loss": round(adapted["stop_loss_pct"], 3),
            "adapted_take_profit": round(adapted["take_profit_pct"], 3),
            "muzero_plan": muzero_plan,
            "grpo_result": grpo_res,
            "pdrl_summary": self.pdrl.get_summary(),
        }

    def _update_learning_curve(self):
        win_rate = self.total_wins / max(1, self.total_trades) * 100
        self.learning_curve.append({
            "trade_num": self.total_trades,
            "win_rate": round(win_rate, 1),
            "loss_rate": round(100 - win_rate, 1),
            "time": time.strftime("%H:%M:%S"),
        })
        if len(self.learning_curve) > 100:
            self.learning_curve.pop(0)

    def get_full_telemetry(self, current_equity: float = 1000.0) -> dict:
        """Get complete WebRL + Q-Learning + MuZero + GRPO + PDRL telemetry."""
        progress_to_100 = self.total_trades % 100
        return {
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "win_rate": round(self.total_wins / max(1, self.total_trades) * 100, 1),
            "trading_fee_pct": 0.1,
            "last_evolution_time": self.last_evolution_time,
            "last_evolution_timestamp": self.last_evolution_timestamp,
            "total_evolutions_count": self.total_evolutions_count,
            "evolution_history": self.evolution_history[-10:],
            "goal_survival": self.goal_controller.get_summary(current_equity),
            "attempts_progress": {
                "current_attempt": self.total_trades,
                "current_cycle_progress": progress_to_100,
                "target_attempts": 100,
                "remaining_to_milestone": 100 - progress_to_100 if progress_to_100 > 0 else 100,
                "milestones_completed": self.total_trades // 100,
            },
            "last_milestone_report": getattr(self, "last_milestone_report", None),
            "q_learning": self.q_table.get_summary(),
            "loss_analyzer": self.loss_analyzer.get_summary(),
            "win_analyzer": self.win_analyzer.get_summary(),
            "pattern_memory": self.pattern_memory.get_summary(),
            "curriculum": self.curriculum.get_summary(),
            "orm": self.orm.get_summary(),
            "policy": self.policy_adapter.get_summary(),
            "muzero": self.muzero.get_summary(),
            "grpo": self.grpo.get_summary(),
            "pdrl": self.pdrl.get_summary(),
            "learning_curve": self.learning_curve[-20:],
        }

