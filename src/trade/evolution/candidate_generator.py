"""Safe, reproducible candidate configuration generation.

Candidates are immutable snapshots. The champion configuration is copied,
never mutated, and each candidate changes one bounded parameter with a stated
hypothesis.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    parent_version: str
    version: str
    config: dict[str, Any]
    hypothesis: str
    seed: int


class CandidateGenerator:
    """Generate bounded one-change hypotheses from an immutable champion."""

    MUTATIONS = (
        # Reward function & PPO hyperparameters
        ("training.reward_turnover_penalty", -0.01, 0.01, "Reduce turnover penalty to test if transaction-cost drag is over-penalized"),
        ("training.reward_drawdown_penalty", -0.05, 0.05, "Adjust drawdown penalty to calibrate downside risk aversion"),
        ("training.reward_risk_penalty", -0.02, 0.02, "Adjust risk penalty to tune model selectivity"),
        ("training.learning_rate", -0.00005, 0.00005, "Tune PPO learning rate for policy optimization stability"),
        ("training.clip_range", -0.02, 0.02, "Adjust PPO clip range to control policy update step size"),
        ("training.ent_coef", -0.002, 0.002, "Calibrate policy entropy coefficient for exploration balance"),

        # Intelligence & Selection parameters
        ("intelligence.minimum_edge", -0.0005, 0.0005, "Adjust minimum net edge threshold to test selectivity"),
        ("intelligence.minimum_signal_confidence", -0.03, 0.03, "Calibrate minimum strategy signal confidence gate"),
        ("intelligence.cost_safety_multiplier", -0.1, 0.1, "Calibrate transaction cost safety margin multiplier"),

        # Risk & Sizing parameters
        ("trading.default_position_size_pct", -1.0, 1.0, "Adjust baseline exposure conservatively while preserving position caps"),
        ("risk.max_position_pct", -2.0, 2.0, "Tune maximum single-position concentration limit"),
        ("risk.max_daily_loss_pct", -0.5, 0.5, "Adjust daily stop-loss threshold to protect portfolio capital"),
        ("risk.caution_drawdown", -0.01, 0.01, "Adjust caution tier drawdown trigger threshold"),
        ("risk.defensive_drawdown", -0.02, 0.02, "Adjust defensive tier drawdown trigger threshold"),

        # Strategy specialist parameters
        ("strategies.trend.min_adx", -2.0, 2.0, "Calibrate minimum ADX filter for trend strategy"),
        ("strategies.breakout.upper_threshold", -0.02, 0.02, "Tune Bollinger Band upper breakout threshold"),
        ("strategies.mean_reversion.rsi_oversold", -2.0, 2.0, "Calibrate RSI oversold threshold for mean reversion"),
        ("strategies.momentum.threshold", -0.002, 0.002, "Calibrate rate-of-change momentum trigger threshold"),
    )

    def __init__(self, seed: int = 0, bounds: dict[str, tuple[float, float]] | None = None):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.bounds = bounds or {
            "training.reward_turnover_penalty": (0.0, 1.0),
            "training.reward_drawdown_penalty": (0.0, 2.0),
            "training.reward_risk_penalty": (0.0, 1.0),
            "training.learning_rate": (0.00001, 0.001),
            "training.clip_range": (0.1, 0.4),
            "training.ent_coef": (0.0, 0.05),
            "intelligence.minimum_edge": (-0.01, 0.01),
            "intelligence.minimum_signal_confidence": (0.45, 0.85),
            "intelligence.cost_safety_multiplier": (1.0, 3.0),
            "trading.default_position_size_pct": (0.1, 20.0),
            "risk.max_position_pct": (5.0, 30.0),
            "risk.max_daily_loss_pct": (1.0, 10.0),
            "risk.caution_drawdown": (0.02, 0.10),
            "risk.defensive_drawdown": (0.05, 0.20),
            "strategies.trend.min_adx": (15.0, 35.0),
            "strategies.breakout.upper_threshold": (0.85, 0.99),
            "strategies.mean_reversion.rsi_oversold": (20.0, 40.0),
            "strategies.momentum.threshold": (0.005, 0.05),
        }

    def generate(self, champion_config: dict[str, Any], parent_version: str, count: int = 1) -> list[Candidate]:
        if count < 0:
            raise ValueError("count must be non-negative")
        candidates: list[Candidate] = []
        for _ in range(count):
            path, low_delta, high_delta, hypothesis = self.rng.choice(self.MUTATIONS)
            current = float(_get_path(champion_config, path, 0.0))
            lower, upper = self.bounds[path]
            value = min(upper, max(lower, current + self.rng.uniform(low_delta, high_delta)))
            config = deepcopy(champion_config)
            _set_path(config, path, value)
            canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(f"{parent_version}:{self.seed}:{canonical}".encode()).hexdigest()[:12]
            candidates.append(Candidate(digest, parent_version, f"{parent_version}-candidate-{digest}", config, hypothesis, self.seed))
        return candidates


def _get_path(config: dict[str, Any], path: str, default: Any) -> Any:
    value: Any = config
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _set_path(config: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
    target[keys[-1]] = value
