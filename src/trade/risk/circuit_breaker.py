"""Circuit breaker: emergency halt mechanism for trading.

Three states:
    CLOSED   → Normal operation, trading allowed
    OPEN     → Trading halted, all orders rejected
    HALF_OPEN → Limited testing, reduced position sizes

The circuit breaker has ABSOLUTE authority — when open, nothing trades.
"""

from __future__ import annotations

import datetime as dt
import logging
import threading

from trade.core.events import CircuitBreakerTripped, event_bus
from trade.core.types import CircuitState

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """Emergency trading halt mechanism.

    Monitors for dangerous conditions and automatically halts trading.
    Supports manual tripping, automatic cooldown recovery, and
    half-open testing state.
    """

    def __init__(
        self,
        cooldown_seconds: float = 3600.0,
        max_consecutive_losses: int = 5,
        max_error_rate: float = 0.5,
        error_window_seconds: float = 300.0,
    ) -> None:
        self._state = CircuitState.CLOSED
        self._cooldown_seconds = cooldown_seconds
        self._max_consecutive_losses = max_consecutive_losses
        self._max_error_rate = max_error_rate
        self._error_window_seconds = error_window_seconds

        self._tripped_at: dt.datetime | None = None
        self._trip_reason: str = ""
        self._consecutive_losses: int = 0
        self._errors: list[dt.datetime] = []
        self._total_checks: int = 0
        self._trip_history: list[dict] = []
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit breaker state."""
        with self._lock:
            # FIX 16: Do NOT auto-recover. Require manual reset.
            # If cooldown has elapsed, emit alert but stay OPEN until explicit reset.
            if self._state == CircuitState.OPEN and self._tripped_at:
                elapsed = (dt.datetime.utcnow() - self._tripped_at).total_seconds()
                if elapsed >= self._cooldown_seconds:
                    # Check if we've already notified
                    if not hasattr(self, "_recovery_notified"):
                        self._recovery_notified = False

                    if not self._recovery_notified:
                        self._recovery_notified = True
                        logger.warning(
                            "⚠️  Circuit breaker cooldown elapsed (%.0fs). "
                            "Call reset() to manually re-enable trading.",
                            elapsed,
                        )
                        # Emit alert event for notification system
                        from trade.core.events import event_bus, CircuitBreakerReadyToReset
                        try:
                            event_bus.publish_sync(
                                CircuitBreakerReadyToReset(
                                    tripped_at=self._tripped_at,
                                    reason=self._trip_reason,
                                )
                            )
                        except Exception:
                            pass  # Event class might not exist yet
            return self._state

    @property
    def is_trading_allowed(self) -> bool:
        """Whether trading is currently permitted."""
        state = self.state
        return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def trip(self, reason: str) -> None:
        """Manually trip the circuit breaker.

        Args:
            reason: Human-readable reason for the trip.
        """
        with self._lock:
            self._state = CircuitState.OPEN
            self._tripped_at = dt.datetime.utcnow()
            self._trip_reason = reason

            self._trip_history.append({
                "timestamp": self._tripped_at.isoformat(),
                "reason": reason,
            })

        logger.critical("🔴 CIRCUIT BREAKER TRIPPED: %s", reason)

        # Emit event
        event_bus.publish_sync(
            CircuitBreakerTripped(
                reason=reason,
                cooldown_seconds=self._cooldown_seconds,
            )
        )

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED."""
        with self._lock:
            old_state = self._state
            self._state = CircuitState.CLOSED
            self._tripped_at = None
            self._trip_reason = ""
            self._consecutive_losses = 0
            self._errors.clear()

        logger.info("🟢 Circuit breaker RESET (%s → CLOSED)", old_state.value)

    def record_loss(self) -> None:
        """Record a losing trade. Trips breaker if consecutive limit exceeded."""
        with self._lock:
            self._consecutive_losses += 1

            if self._consecutive_losses >= self._max_consecutive_losses:
                reason = (
                    f"Consecutive losses: {self._consecutive_losses} "
                    f"(limit: {self._max_consecutive_losses})"
                )
                # Release lock before tripping (trip acquires lock)
                self._state = CircuitState.OPEN
                self._tripped_at = dt.datetime.utcnow()
                self._trip_reason = reason
                self._trip_history.append({
                    "timestamp": self._tripped_at.isoformat(),
                    "reason": reason,
                })

        if self._consecutive_losses >= self._max_consecutive_losses:
            logger.critical("🔴 CIRCUIT BREAKER TRIPPED: %s", reason)
            event_bus.publish_sync(
                CircuitBreakerTripped(
                    reason=reason,
                    cooldown_seconds=self._cooldown_seconds,
                )
            )

    def record_win(self) -> None:
        """Record a winning trade. Resets consecutive loss counter."""
        with self._lock:
            self._consecutive_losses = 0

            # If in HALF_OPEN and we get a win, close the circuit
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info("🟢 Circuit breaker CLOSED after successful half-open trade")

    def record_error(self) -> None:
        """Record a system error. Trips if error rate exceeds threshold."""
        now = dt.datetime.utcnow()
        with self._lock:
            self._errors.append(now)
            self._total_checks += 1

            # Clean old errors outside window
            cutoff = now - dt.timedelta(seconds=self._error_window_seconds)
            self._errors = [e for e in self._errors if e > cutoff]

            # Check error rate
            if self._total_checks >= 10:  # Minimum sample size
                recent_errors = len(self._errors)
                error_rate = recent_errors / min(self._total_checks, 100)

                if error_rate > self._max_error_rate:
                    reason = f"Error rate: {error_rate:.2%} (limit: {self._max_error_rate:.2%})"
                    self._state = CircuitState.OPEN
                    self._tripped_at = now
                    self._trip_reason = reason

    def get_status(self) -> dict:
        """Return current circuit breaker status."""
        return {
            "state": self.state.value,
            "trading_allowed": self.is_trading_allowed,
            "consecutive_losses": self._consecutive_losses,
            "recent_errors": len(self._errors),
            "tripped_at": self._tripped_at.isoformat() if self._tripped_at else None,
            "trip_reason": self._trip_reason,
            "cooldown_seconds": self._cooldown_seconds,
            "total_trips": len(self._trip_history),
        }
