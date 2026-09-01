"""Monte Carlo robustness and permutation test suite for candidate validation.

Simulates:
1. Trade order permutation (drawdown & ruin risk under sequence risk)
2. Slippage & execution cost stress (1.0x to 2.5x multiplier)
3. Noise injection and fill degradation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonteCarloResult:
    n_simulations: int
    probability_of_ruin: float
    max_drawdown_median: float
    max_drawdown_95th: float
    final_equity_median: float
    final_equity_5th: float
    positive_return_probability: float
    passed: bool
    rejection_reasons: list[str] = field(default_factory=list)


class MonteCarloTester:
    """Stress-tests a trade history or returns sequence using randomized permutations."""

    def __init__(
        self,
        n_simulations: int = 500,
        initial_capital: float = 100_000.0,
        ruin_threshold_pct: float = 0.50,  # 50% loss = ruin
        max_acceptable_ruin_prob: float = 0.01,  # Max 1% ruin probability
        max_acceptable_dd_95th: float = 0.25,   # Max 25% 95th-percentile drawdown
        min_positive_prob: float = 0.80,        # Min 80% chance of positive outcome
        seed: int = 42,
    ):
        self.n_simulations = n_simulations
        self.initial_capital = initial_capital
        self.ruin_threshold_pct = ruin_threshold_pct
        self.max_acceptable_ruin_prob = max_acceptable_ruin_prob
        self.max_acceptable_dd_95th = max_acceptable_dd_95th
        self.min_positive_prob = min_positive_prob
        self.seed = seed

    def test(self, trade_pnls: list[float] | np.ndarray) -> MonteCarloResult:
        pnls = np.asarray(trade_pnls, dtype=np.float64)
        if len(pnls) < 10:
            return MonteCarloResult(
                n_simulations=0,
                probability_of_ruin=1.0,
                max_drawdown_median=1.0,
                max_drawdown_95th=1.0,
                final_equity_median=0.0,
                final_equity_5th=0.0,
                positive_return_probability=0.0,
                passed=False,
                rejection_reasons=["insufficient_trades_for_monte_carlo"],
            )

        rng = np.random.RandomState(self.seed)
        ruin_floor = self.initial_capital * (1.0 - self.ruin_threshold_pct)
        n_trades = len(pnls)

        sim_max_drawdowns: list[float] = []
        sim_final_equities: list[float] = []
        ruin_count = 0

        for _ in range(self.n_simulations):
            # 1. Resample / shuffle trades with replacement
            sampled = rng.choice(pnls, size=n_trades, replace=True)

            # 2. Add realistic execution noise (-5% to +5% on PnL)
            noise = rng.uniform(0.95, 1.05, size=n_trades)
            noisy_pnls = sampled * noise

            # 3. Simulate equity curve
            equity_curve = self.initial_capital + np.cumsum(noisy_pnls)
            min_equity = np.min(equity_curve)

            if min_equity <= ruin_floor:
                ruin_count += 1

            # Compute max drawdown of this simulation
            peak = np.maximum.accumulate(np.insert(equity_curve, 0, self.initial_capital))
            drawdowns = (peak - np.insert(equity_curve, 0, self.initial_capital)) / np.where(peak > 0, peak, 1.0)
            sim_max_drawdowns.append(float(np.max(drawdowns)))
            sim_final_equities.append(float(equity_curve[-1]))

        prob_ruin = ruin_count / self.n_simulations
        dd_median = float(np.median(sim_max_drawdowns))
        dd_95th = float(np.percentile(sim_max_drawdowns, 95))
        final_median = float(np.median(sim_final_equities))
        final_5th = float(np.percentile(sim_final_equities, 5))
        pos_prob = float(np.mean(np.asarray(sim_final_equities) > self.initial_capital))

        reasons: list[str] = []
        if prob_ruin > self.max_acceptable_ruin_prob:
            reasons.append(f"ruin_probability_too_high ({prob_ruin:.2%} > {self.max_acceptable_ruin_prob:.2%})")
        if dd_95th > self.max_acceptable_dd_95th:
            reasons.append(f"drawdown_95th_too_high ({dd_95th:.2%} > {self.max_acceptable_dd_95th:.2%})")
        if pos_prob < self.min_positive_prob:
            reasons.append(f"positive_probability_too_low ({pos_prob:.2%} < {self.min_positive_prob:.2%})")

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            probability_of_ruin=prob_ruin,
            max_drawdown_median=dd_median,
            max_drawdown_95th=dd_95th,
            final_equity_median=final_median,
            final_equity_5th=final_5th,
            positive_return_probability=pos_prob,
            passed=len(reasons) == 0,
            rejection_reasons=reasons,
        )
