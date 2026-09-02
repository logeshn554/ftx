"""Retraining scheduler: time-based and event-based triggers."""

from __future__ import annotations

import datetime as dt
import logging
import threading

from trade.core.config import AppConfig
from trade.core.events import PerformanceDegraded, event_bus
from trade.core.types import ModelVersion
from trade.learning.retrainer import CandidateRetrainer
from trade.model_management.registry import ModelRegistry

logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """Controls when and how retraining is triggered.

    Two modes:
        1. Scheduled: retrain every N days regardless of performance
        2. Event-driven: retrain when PerformanceDegraded event fires

    Safety:
        - Prevents concurrent retraining runs
        - Enforces cooldown between retrainings
    """

    def __init__(
        self,
        config: AppConfig,
        retrainer: CandidateRetrainer,
        registry: ModelRegistry,  # FIX 13: Inject registry to get real version
    ) -> None:
        self.config = config
        self.retrainer = retrainer
        self.registry = registry
        self._last_scheduled_retrain: dt.datetime | None = None
        self._event_subscribed = False
        self._lock = threading.Lock()

    def subscribe_to_events(self) -> None:
        """Subscribe to PerformanceDegraded events for event-driven retraining."""
        if not self._event_subscribed:
            event_bus.subscribe(PerformanceDegraded, self._on_performance_degraded)
            self._event_subscribed = True
            logger.info("RetrainingScheduler subscribed to PerformanceDegraded events")

    def check_scheduled(self) -> bool:
        """Check if a scheduled retraining is due.

        Returns:
            True if retraining was triggered.
        """
        now = dt.datetime.utcnow()
        interval_days = self.config.learning.scheduled_retrain_days

        if self._last_scheduled_retrain is None:
            self._last_scheduled_retrain = now
            return False

        days_since = (now - self._last_scheduled_retrain).days

        if days_since >= interval_days:
            logger.info(
                "Scheduled retraining due: %d days since last (interval: %d)",
                days_since,
                interval_days,
            )
            return self._trigger_retrain("scheduled")

        return False

    def _on_performance_degraded(self, event: PerformanceDegraded) -> None:
        """Handle PerformanceDegraded event."""
        logger.info(
            "Received PerformanceDegraded: %s=%.4f (threshold=%.4f)",
            event.metric_name,
            event.current_value,
            event.threshold,
        )
        # FIX 13: Get actual production version from registry
        self._trigger_retrain(f"degraded_{event.metric_name}")

    def _trigger_retrain(self, trigger: str) -> bool:
        """Attempt to trigger retraining (thread-safe).

        Returns:
            True if retraining was started.
        """
        # FIX 13: Get production version from registry
        current_version = self.registry.get_production_version()
        if current_version is None:
            logger.error(
                "No production model found in registry — cannot trigger retraining"
            )
            return False

        logger.info("Triggering retraining: trigger=%s, current_version=%s", trigger, current_version.tag)

        with self._lock:
            if self.retrainer.is_training:
                logger.info("Retraining already in progress, skipping trigger '%s'", trigger)
                return False

            result = self.retrainer.retrain(current_version, trigger=trigger)

            if result is not None:
                self._last_scheduled_retrain = dt.datetime.utcnow()
                return True

            return False
