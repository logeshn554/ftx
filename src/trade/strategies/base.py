from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class Signal:
    side: str
    confidence: float
    expected_move: float
    stop_distance: float
    target_distance: float
    strategy: str

class Strategy(Protocol):
    name: str
    def signal(self, indicators: dict) -> Signal: ...

def abstain(name: str) -> Signal:
    return Signal("HOLD", 0.0, 0.0, 0.0, 0.0, name)
