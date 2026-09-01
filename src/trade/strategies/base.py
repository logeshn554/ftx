from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Signal:
    side: str  # "BUY" | "SELL" | "HOLD"
    confidence: float
    expected_move: float
    stop_distance: float
    target_distance: float
    strategy: str
    reason: str = ""


class Strategy(Protocol):
    name: str

    def signal(self, indicators: dict) -> Signal: ...


def abstain(name: str, reason: str = "no_signal") -> Signal:
    return Signal(
        side="HOLD",
        confidence=0.0,
        expected_move=0.0,
        stop_distance=0.0,
        target_distance=0.0,
        strategy=name,
        reason=reason,
    )
