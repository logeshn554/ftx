"""Safe, reproducible candidate configuration generation.

Candidates are immutable snapshots.  The champion configuration is copied,
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
        ("training.reward_turnover_penalty", -0.01, 0.01, "Reduce turnover because transaction-cost drag may be excessive"),
        ("training.reward_drawdown_penalty", -0.05, 0.05, "Adjust drawdown aversion because downside risk may be under-penalized"),
        ("trading.default_position_size_pct", -1.0, 1.0, "Adjust exposure conservatively while preserving the position-size cap"),
        ("intelligence.minimum_edge", -0.0005, 0.0005, "Adjust the minimum edge threshold to test cost-aware selectivity"),
    )

    def __init__(self, seed: int = 0, bounds: dict[str, tuple[float, float]] | None = None):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.bounds = bounds or {
            "training.reward_turnover_penalty": (0.0, 1.0),
            "training.reward_drawdown_penalty": (0.0, 2.0),
            "trading.default_position_size_pct": (0.1, 20.0),
            "intelligence.minimum_edge": (-0.01, 0.01),
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
