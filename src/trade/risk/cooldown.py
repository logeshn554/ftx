"""Anti-churn controls and trading frequency governors.

Prevents excessive fee burn, rapid over-trading, and churn
by enforcing minimum holding times, inter-trade intervals,
and rolling turnover/trade-count limits.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class CooldownConfig:
    min_inter_trade_bars: int = 2          # Minimum bars between any two trade entries
    min_inter_trade_seconds: float = 3.0   # Minimum real-time seconds between entries
    min_holding_bars: int = 2              # Minimum bars to hold a position unless stopped out
    max_trades_per_window: int = 6         # Maximum trades in the sliding window
    window_bars: int = 50                  # Length of the sliding window in bars
    max_window_turnover_ratio: float = 3.0 # Max cumulative turnover in window as multiple of equity


class CooldownController:
    """Monitors and regulates trading velocity to eliminate churning."""

    def __init__(self, config: CooldownConfig | None = None):
        self.config = config or CooldownConfig()
        self._last_entry_bar: int = -999
        self._last_entry_time: float = 0.0
        self._last_close_bar: int = -999
        self._current_bar: int = 0
        self._trade_history: deque[Tuple[int, float]] = deque()  # (bar_index, notional)

    def advance_bar(self, current_bar: int | None = None) -> None:
        """Advance the internal bar counter and prune stale window entries."""
        if current_bar is not None:
            self._current_bar = current_bar
        else:
            self._current_bar += 1

        # Prune trade history outside sliding window
        cutoff = self._current_bar - self.config.window_bars
        while self._trade_history and self._trade_history[0][0] < cutoff:
            self._trade_history.popleft()

    def can_enter(self, equity: float) -> Tuple[bool, str]:
        """Check whether a new trade entry is allowed under anti-churn rules."""
        # 1. Check minimum bar interval since last trade
        bars_since_last = self._current_bar - self._last_entry_bar
        if bars_since_last < self.config.min_inter_trade_bars:
            return False, f"INTER_TRADE_COOLDOWN (waited {bars_since_last}/{self.config.min_inter_trade_bars} bars)"

        # 2. Check minimum wall-clock seconds interval
        elapsed_seconds = time.time() - self._last_entry_time
        if elapsed_seconds < self.config.min_inter_trade_seconds:
            return False, f"WALLCLOCK_COOLDOWN (waited {elapsed_seconds:.1f}s / {self.config.min_inter_trade_seconds}s)"

        # 3. Check sliding window trade count limit
        recent_trades = len(self._trade_history)
        if recent_trades >= self.config.max_trades_per_window:
            return False, f"WINDOW_TRADE_LIMIT ({recent_trades}/{self.config.max_trades_per_window} in {self.config.window_bars} bars)"

        # 4. Check sliding window turnover limit
        if equity > 0:
            recent_turnover = sum(notional for _, notional in self._trade_history)
            max_turnover = equity * self.config.max_window_turnover_ratio
            if recent_turnover >= max_turnover:
                return False, f"TURNOVER_LIMIT (${recent_turnover:,.2f} >= ${max_turnover:,.2f} cap)"

        return True, "OK"

    def record_entry(self, notional: float = 0.0, bar_index: int | None = None) -> None:
        """Record a successful trade entry."""
        if bar_index is not None:
            self._current_bar = bar_index
        self._last_entry_bar = self._current_bar
        self._last_entry_time = time.time()
        self._trade_history.append((self._current_bar, float(notional)))

    def record_close(self, notional: float = 0.0, bar_index: int | None = None) -> None:
        """Record a trade close."""
        if bar_index is not None:
            self._current_bar = bar_index
        self._last_close_bar = self._current_bar
        self._trade_history.append((self._current_bar, float(notional)))

    def reset(self) -> None:
        """Reset all cooldown timers and history."""
        self._last_entry_bar = -999
        self._last_entry_time = 0.0
        self._last_close_bar = -999
        self._trade_history.clear()
