"""Lightweight async event bus for inter-component signaling.

Components publish events (e.g. TradeExecuted, PerformanceDegraded) and
other components subscribe to react — without direct coupling.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """Base event. All events carry a timestamp and optional payload."""

    timestamp: dt.datetime = field(default_factory=dt.datetime.utcnow)


@dataclass(frozen=True)
class TradeExecuted(Event):
    """Emitted after a trade is filled."""

    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float = 0.0
    pnl: float = 0.0


@dataclass(frozen=True)
class PerformanceDegraded(Event):
    """Emitted when the performance monitor detects metric degradation."""

    metric_name: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    window_days: int = 0


@dataclass(frozen=True)
class ModelPromoted(Event):
    """Emitted when a candidate model is promoted to production."""

    old_version: str = ""
    new_version: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ModelRolledBack(Event):
    """Emitted when production model is rolled back."""

    from_version: str = ""
    to_version: str = ""
    reason: str = ""


@dataclass(frozen=True)
class RiskBreached(Event):
    """Emitted when a risk limit is breached."""

    limit_name: str = ""
    current_value: float = 0.0
    limit_value: float = 0.0
    action_taken: str = ""


@dataclass(frozen=True)
class CircuitBreakerTripped(Event):
    """Emitted when the circuit breaker opens."""

    reason: str = ""
    cooldown_seconds: float = 0.0


@dataclass(frozen=True)
class RetrainingStarted(Event):
    """Emitted when candidate retraining begins."""

    trigger: str = ""
    model_version: str = ""


@dataclass(frozen=True)
class RetrainingCompleted(Event):
    """Emitted when candidate retraining finishes."""

    model_version: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

# Type alias for handlers: sync or async callables accepting an Event
EventHandler = Callable[[Event], Any] | Callable[[Event], Coroutine[Any, Any, Any]]


class EventBus:
    """Simple publish/subscribe event bus.

    Supports both sync and async handlers. Handlers are invoked in
    registration order. Exceptions in handlers are logged but do not
    propagate to the publisher.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[EventHandler]] = {}

    def subscribe(self, event_type: type[Event], handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: type[Event], handler: EventHandler) -> None:
        """Remove a previously registered handler."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to all registered handlers.

        Async handlers are awaited; sync handlers are called directly.
        """
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "Error in event handler %s for %s",
                    getattr(handler, "__name__", handler),
                    event_type.__name__,
                )

    def publish_sync(self, event: Event) -> None:
        """Publish an event synchronously (for non-async contexts).

        Only invokes sync handlers; async handlers are skipped with a warning.
        """
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        for handler in handlers:
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    logger.warning(
                        "Async handler %s skipped in sync publish for %s",
                        getattr(handler, "__name__", handler),
                        event_type.__name__,
                    )
                    result.close()  # Prevent coroutine-never-awaited warning
            except Exception:
                logger.exception(
                    "Error in event handler %s for %s",
                    getattr(handler, "__name__", handler),
                    event_type.__name__,
                )


# Global singleton event bus — components import and use this instance
event_bus = EventBus()
