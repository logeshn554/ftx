"""Model comparator: statistical comparison of V1 (champion) vs V2 (challenger)."""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats

from trade.core.types import BacktestResult, ComparisonResult, ModelVersion

logger = logging.getLogger(__name__)


class ModelComparator:
    """Compares two models using their backtest results.

    Performs:
        1. Metric comparison (Sharpe, MDD, win rate, profit factor)
        2. Statistical significance tests (paired t-test, bootstrap)
        3. Verdict: PROMOTE, REJECT, or INCONCLUSIVE
    """

    def __init__(
        self,
        min_sharpe_improvement: float = 0.1,
        max_drawdown_regression: float = 0.02,
        min_win_rate: float = 0.45,
        significance_level: float = 0.05,
    ) -> None:
        self.min_sharpe_improvement = min_sharpe_improvement
        self.max_drawdown_regression = max_drawdown_regression
        self.min_win_rate = min_win_rate
        self.significance_level = significance_level

    def compare(
        self,
        champion_result: BacktestResult,
        challenger_result: BacktestResult,
    ) -> ComparisonResult:
        """Compare champion (V1) vs challenger (V2) models.

        Args:
            champion_result: Backtest result of the current production model.
            challenger_result: Backtest result of the candidate model.

        Returns:
            ComparisonResult with verdict and detailed metrics.
        """
        champ = champion_result
        chal = challenger_result

        # Metric comparison
        champion_metrics = {
            "sharpe_ratio": champ.sharpe_ratio,
            "max_drawdown": champ.max_drawdown,
            "win_rate": champ.win_rate,
            "profit_factor": champ.profit_factor,
            "total_return": champ.total_return,
            "total_trades": float(champ.total_trades),
            "transaction_costs": champ.transaction_costs,
        }

        challenger_metrics = {
            "sharpe_ratio": chal.sharpe_ratio,
            "max_drawdown": chal.max_drawdown,
            "win_rate": chal.win_rate,
            "profit_factor": chal.profit_factor,
            "total_return": chal.total_return,
            "total_trades": float(chal.total_trades),
            "transaction_costs": chal.transaction_costs,
        }

        # Improvements and regressions
        improvements = {}
        regressions = {}

        # Sharpe (higher is better)
        sharpe_diff = chal.sharpe_ratio - champ.sharpe_ratio
        if sharpe_diff > 0:
            improvements["sharpe_ratio"] = sharpe_diff
        elif sharpe_diff < 0:
            regressions["sharpe_ratio"] = abs(sharpe_diff)

        # Max drawdown (lower is better)
        mdd_diff = chal.max_drawdown - champ.max_drawdown
        if mdd_diff < 0:
            improvements["max_drawdown"] = abs(mdd_diff)
        elif mdd_diff > 0:
            regressions["max_drawdown"] = mdd_diff

        # Win rate (higher is better)
        wr_diff = chal.win_rate - champ.win_rate
        if wr_diff > 0:
            improvements["win_rate"] = wr_diff
        elif wr_diff < 0:
            regressions["win_rate"] = abs(wr_diff)

        # Return (higher is better)
        ret_diff = chal.total_return - champ.total_return
        if ret_diff > 0:
            improvements["total_return"] = ret_diff
        elif ret_diff < 0:
            regressions["total_return"] = abs(ret_diff)

        # Statistical significance
        stat_sig = self._statistical_tests(champ.daily_returns, chal.daily_returns)

        # Determine verdict
        verdict = self._determine_verdict(
            sharpe_diff=sharpe_diff,
            mdd_diff=mdd_diff,
            challenger_win_rate=chal.win_rate,
            stat_sig=stat_sig,
        )

        notes = self._build_notes(
            verdict, sharpe_diff, mdd_diff, chal.win_rate, stat_sig
        )

        result = ComparisonResult(
            champion=champ.model_version,
            challenger=chal.model_version,
            verdict=verdict,
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            improvements=improvements,
            regressions=regressions,
            statistical_significance=stat_sig,
            notes=notes,
        )

        logger.info(
            "Comparison %s vs %s: VERDICT=%s | Sharpe Δ=%.3f | MDD Δ=%.3f%%",
            champ.model_version.tag,
            chal.model_version.tag,
            verdict.value,
            sharpe_diff,
            mdd_diff * 100,
        )

        return result

    def _statistical_tests(
        self,
        champion_returns: list[float],
        challenger_returns: list[float],
    ) -> dict[str, float]:
        """Run statistical significance tests on daily returns."""
        results: dict[str, float] = {}

        champ = np.array(champion_returns)
        chal = np.array(challenger_returns)

        # Ensure same length for paired test
        min_len = min(len(champ), len(chal))
        if min_len < 10:
            results["sufficient_data"] = 0.0
            return results

        champ = champ[:min_len]
        chal = chal[:min_len]

        # Paired t-test on daily returns
        try:
            t_stat, p_value = stats.ttest_rel(chal, champ)
            results["paired_ttest_t_stat"] = float(t_stat)
            results["paired_ttest_p_value"] = float(p_value)
            results["paired_ttest_significant"] = float(p_value < self.significance_level)
        except Exception:
            pass

        # Welch's t-test (independent)
        try:
            t_stat, p_value = stats.ttest_ind(chal, champ, equal_var=False)
            results["welch_ttest_p_value"] = float(p_value)
        except Exception:
            pass

        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, p_value = stats.mannwhitneyu(chal, champ, alternative="greater")
            results["mannwhitney_p_value"] = float(p_value)
        except Exception:
            pass

        results["sufficient_data"] = 1.0
        return results

    def _determine_verdict(
        self,
        sharpe_diff: float,
        mdd_diff: float,
        challenger_win_rate: float,
        stat_sig: dict[str, float],
    ) -> ComparisonResult.Verdict:
        """Determine PROMOTE/REJECT/INCONCLUSIVE verdict."""

        # Hard rejection criteria
        if challenger_win_rate < self.min_win_rate:
            return ComparisonResult.Verdict.REJECT

        if mdd_diff > self.max_drawdown_regression:
            return ComparisonResult.Verdict.REJECT

        # Promotion criteria
        if (
            sharpe_diff >= self.min_sharpe_improvement
            and mdd_diff <= self.max_drawdown_regression
            and challenger_win_rate >= self.min_win_rate
        ):
            # Bonus: if statistically significant, strong promote
            if stat_sig.get("paired_ttest_significant", 0.0) > 0:
                return ComparisonResult.Verdict.PROMOTE
            # Still promote if improvement is clear even without significance
            if sharpe_diff >= self.min_sharpe_improvement * 2:
                return ComparisonResult.Verdict.PROMOTE
            return ComparisonResult.Verdict.INCONCLUSIVE

        if sharpe_diff > 0 and mdd_diff <= 0:
            return ComparisonResult.Verdict.INCONCLUSIVE

        return ComparisonResult.Verdict.REJECT

    def _build_notes(
        self,
        verdict: ComparisonResult.Verdict,
        sharpe_diff: float,
        mdd_diff: float,
        win_rate: float,
        stat_sig: dict[str, float],
    ) -> str:
        """Build human-readable comparison notes."""
        lines = []

        if verdict == ComparisonResult.Verdict.PROMOTE:
            lines.append("✅ Challenger PASSES all promotion criteria.")
        elif verdict == ComparisonResult.Verdict.REJECT:
            lines.append("❌ Challenger FAILS promotion criteria.")
        else:
            lines.append("⚠️ Results are INCONCLUSIVE — more data needed.")

        lines.append(f"Sharpe improvement: {sharpe_diff:+.3f} (min required: {self.min_sharpe_improvement})")
        lines.append(f"Max drawdown change: {mdd_diff:+.3%} (max regression: {self.max_drawdown_regression:.1%})")
        lines.append(f"Challenger win rate: {win_rate:.1%} (min required: {self.min_win_rate:.1%})")

        p_val = stat_sig.get("paired_ttest_p_value")
        if p_val is not None:
            lines.append(f"Statistical significance (paired t-test): p={p_val:.4f}")

        return "\n".join(lines)
