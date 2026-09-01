"""Performance monitor: detects metric degradation in the production agent.

Watches rolling performance metrics and emits PerformanceDegraded events
when thresholds are breached, triggering the candidate retraining pipeline.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from trade.core.events import PerformanceDegraded, event_bus

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """A point-in-time snapshot of agent performance metrics."""

    timestamp: dt.datetime = field(default_factory=dt.datetime.utcnow)
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    portfolio_value: float = 0.0


class PerformanceMonitor:
    """Tracks rolling performance of the production agent.

    Computes trailing metrics over configurable windows and detects
    when performance drops below thresholds.
    """

    def __init__(
        self,
        sharpe_threshold: float = 0.5,
        window_days: int = 30,
        check_interval_hours: float = 1.0,
    ) -> None:
        self.sharpe_threshold = sharpe_threshold
        self.window_days = window_days
        self.check_interval_hours = check_interval_hours

        self._daily_returns: deque[float] = deque(maxlen=365)
        self._snapshots: list[PerformanceSnapshot] = []
        self._last_check: dt.datetime | None = None
        self._degraded = False
        self._peak_value: float = 0.0
        self._wins = 0
        self._losses = 0

    def record_daily_return(self, daily_return: float, portfolio_value: float) -> None:
        """Record a daily return observation.

        Args:
            daily_return: The portfolio return for this period.
            portfolio_value: Current total portfolio value.
        """
        self._daily_returns.append(daily_return)
        self._peak_value = max(self._peak_value, portfolio_value)

        if daily_return > 0:
            self._wins += 1
        elif daily_return < 0:
            self._losses += 1

        # Compute current metrics
        snapshot = self._compute_snapshot(portfolio_value)
        self._snapshots.append(snapshot)

        # Check for degradation
        self._check_degradation(snapshot)

    def record_trade(self, pnl: float) -> None:
        """Record a trade result for win rate tracking."""
        if pnl > 0:
            self._wins += 1
        elif pnl < 0:
            self._losses += 1

    def _compute_snapshot(self, portfolio_value: float) -> PerformanceSnapshot:
        """Compute current performance metrics."""
        returns = list(self._daily_returns)
        window_returns = returns[-self.window_days:] if len(returns) >= self.window_days else returns

        # Sharpe ratio (annualized)
        if len(window_returns) >= 5:
            mean_ret = np.mean(window_returns)
            std_ret = np.std(window_returns)
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 1e-8 else 0.0
        else:
            sharpe = 0.0

        # Max drawdown
        if self._peak_value > 0:
            drawdown = (self._peak_value - portfolio_value) / self._peak_value
        else:
            drawdown = 0.0

        # Win rate
        total_trades = self._wins + self._losses
        win_rate = self._wins / total_trades if total_trades > 0 else 0.0

        # Cumulative return
        if len(returns) > 0:
            cum_return = float(np.prod([1 + r for r in returns]) - 1)
        else:
            cum_return = 0.0

        return PerformanceSnapshot(
            daily_return=returns[-1] if returns else 0.0,
            cumulative_return=cum_return,
            sharpe_ratio=float(sharpe),
            max_drawdown=float(drawdown),
            win_rate=float(win_rate),
            portfolio_value=portfolio_value,
        )

    def _check_degradation(self, snapshot: PerformanceSnapshot) -> None:
        """Check if performance has degraded below thresholds."""
        now = dt.datetime.utcnow()

        # Rate-limit checks
        if self._last_check and (now - self._last_check).total_seconds() < self.check_interval_hours * 3600:
            return

        self._last_check = now

        # Need minimum data
        if len(self._daily_returns) < self.window_days:
            return

        # Check Sharpe threshold
        if snapshot.sharpe_ratio < self.sharpe_threshold:
            if not self._degraded:
                self._degraded = True
                logger.warning(
                    "⚠️ Performance degraded: Sharpe %.2f < threshold %.2f (window: %dd)",
                    snapshot.sharpe_ratio,
                    self.sharpe_threshold,
                    self.window_days,
                )
                event_bus.publish_sync(
                    PerformanceDegraded(
                        metric_name="sharpe_ratio",
                        current_value=snapshot.sharpe_ratio,
                        threshold=self.sharpe_threshold,
                        window_days=self.window_days,
                    )
                )
        else:
            if self._degraded:
                logger.info(
                    "Performance recovered: Sharpe %.2f >= threshold %.2f",
                    snapshot.sharpe_ratio,
                    self.sharpe_threshold,
                )
            self._degraded = False

    def get_current_metrics(self) -> dict[str, float]:
        """Return the latest performance metrics."""
        if not self._snapshots:
            return {}

        s = self._snapshots[-1]
        return {
            "sharpe_ratio": s.sharpe_ratio,
            "max_drawdown": s.max_drawdown,
            "win_rate": s.win_rate,
            "cumulative_return": s.cumulative_return,
            "portfolio_value": s.portfolio_value,
            "total_observations": len(self._daily_returns),
            "is_degraded": self._degraded,
        }

    @property
    def is_degraded(self) -> bool:
        return self._degraded
