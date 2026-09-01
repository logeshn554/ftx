"""Tests for model comparator."""

import pytest

from trade.core.types import BacktestResult, ComparisonResult, ModelVersion
from trade.validation.comparator import ModelComparator


class TestModelComparator:
    """Test V1 vs V2 comparison logic."""

    def _make_result(self, sharpe, mdd, win_rate, total_return, version_minor=1):
        return BacktestResult(
            model_version=ModelVersion(major=0, minor=version_minor, patch=0),
            start_date=__import__("datetime").date(2024, 1, 1),
            end_date=__import__("datetime").date(2024, 12, 31),
            initial_capital=100_000,
            final_capital=100_000 * (1 + total_return),
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            win_rate=win_rate,
            profit_factor=1.5,
            total_trades=100,
            daily_returns=[0.001] * 252,
        )

    def test_promote_when_better(self):
        """Challenger with better metrics gets PROMOTE."""
        comparator = ModelComparator(
            min_sharpe_improvement=0.1,
            max_drawdown_regression=0.02,
            min_win_rate=0.45,
        )

        champion = self._make_result(sharpe=1.0, mdd=0.10, win_rate=0.50, total_return=0.15, version_minor=1)
        challenger = self._make_result(sharpe=1.5, mdd=0.08, win_rate=0.55, total_return=0.25, version_minor=2)

        result = comparator.compare(champion, challenger)
        assert result.verdict == ComparisonResult.Verdict.PROMOTE

    def test_reject_when_worse(self):
        """Challenger with worse metrics gets REJECT."""
        comparator = ModelComparator()

        champion = self._make_result(sharpe=1.5, mdd=0.08, win_rate=0.55, total_return=0.20, version_minor=1)
        challenger = self._make_result(sharpe=0.5, mdd=0.20, win_rate=0.40, total_return=-0.05, version_minor=2)

        result = comparator.compare(champion, challenger)
        assert result.verdict == ComparisonResult.Verdict.REJECT

    def test_reject_low_win_rate(self):
        """Reject even with better Sharpe if win rate is too low."""
        comparator = ModelComparator(min_win_rate=0.45)

        champion = self._make_result(sharpe=1.0, mdd=0.10, win_rate=0.50, total_return=0.15, version_minor=1)
        challenger = self._make_result(sharpe=1.5, mdd=0.08, win_rate=0.40, total_return=0.25, version_minor=2)

        result = comparator.compare(champion, challenger)
        assert result.verdict == ComparisonResult.Verdict.REJECT

    def test_reject_excessive_drawdown_regression(self):
        """Reject if drawdown regresses too much."""
        comparator = ModelComparator(max_drawdown_regression=0.02)

        champion = self._make_result(sharpe=1.0, mdd=0.10, win_rate=0.50, total_return=0.15, version_minor=1)
        challenger = self._make_result(sharpe=1.5, mdd=0.15, win_rate=0.55, total_return=0.25, version_minor=2)

        result = comparator.compare(champion, challenger)
        assert result.verdict == ComparisonResult.Verdict.REJECT

    def test_improvements_tracked(self):
        """Improvements are correctly identified."""
        comparator = ModelComparator()

        champion = self._make_result(sharpe=1.0, mdd=0.10, win_rate=0.50, total_return=0.15, version_minor=1)
        challenger = self._make_result(sharpe=1.5, mdd=0.08, win_rate=0.55, total_return=0.25, version_minor=2)

        result = comparator.compare(champion, challenger)
        assert "sharpe_ratio" in result.improvements
        assert "max_drawdown" in result.improvements
        assert "win_rate" in result.improvements
