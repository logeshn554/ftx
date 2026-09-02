"""Automatic daily/weekly risk reset scheduler."""

from __future__ import annotations

import datetime as dt
import logging
import threading
import time

import pytz

from trade.risk.engine import RiskEngine

logger = logging.getLogger(__name__)


class DailyResetScheduler:
    """Automatically reset risk engine daily and weekly.

    Resets at a configured time (default 09:30 US/Eastern) every trading day.
    Also resets weekly limits on Monday mornings.
    """

    def __init__(
        self,
        risk_engine: RiskEngine,
        starting_equity: float = 100_000.0,
        reset_time: str = "09:30",
        timezone: str = "US/Eastern",
    ) -> None:
        self._risk_engine = risk_engine
        self._starting_equity = starting_equity
        self._reset_time = reset_time
        self._timezone = pytz.timezone(timezone)
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        """Start the scheduler in a background thread."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(
            "Daily reset scheduler started (reset time: %s %s)",
            self._reset_time,
            self._timezone.zone,
        )

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            logger.info("Daily reset scheduler stopped")

    def _loop(self) -> None:
        """Background loop that fires reset at the configured time."""
        while self._running:
            try:
                now = dt.datetime.now(self._timezone)
                h, m = map(int, self._reset_time.split(":"))
                target = now.replace(hour=h, minute=m, second=0, microsecond=0)

                # If we're past the reset time today, schedule for tomorrow
                if now >= target:
                    target += dt.timedelta(days=1)

                sleep_seconds = (target - now).total_seconds()

                # Sleep in small chunks to allow for graceful shutdown
                chunks = int(sleep_seconds / 5) + 1
                for _ in range(chunks):
                    if not self._running:
                        return
                    time.sleep(min(5, sleep_seconds / chunks))

                # Execute reset
                self._risk_engine.reset_daily(self._starting_equity)

                # Also reset weekly on Monday
                if dt.datetime.now(self._timezone).weekday() == 0:
                    self._risk_engine.reset_weekly(self._starting_equity)
                    logger.info("Weekly reset executed on Monday")

            except Exception as e:
                logger.error("Error in daily reset scheduler: %s", e)
                time.sleep(60)  # Wait 1 minute before retrying
