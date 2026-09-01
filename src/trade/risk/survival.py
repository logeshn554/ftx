"""Portfolio survival state machine; HALTED always blocks new entries."""
from __future__ import annotations
from enum import Enum


class SurvivalState(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    DEFENSIVE = "DEFENSIVE"
    HALTED = "HALTED"


class SurvivalController:
    def __init__(self, caution_drawdown: float = .05, defensive_drawdown: float = .10,
                 halt_drawdown: float = .20, max_consecutive_losses: int = 5):
        self.caution_drawdown, self.defensive_drawdown, self.halt_drawdown = caution_drawdown, defensive_drawdown, halt_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.state = SurvivalState.NORMAL

    def update(self, drawdown: float, consecutive_losses: int = 0, daily_loss: float = 0.0,
               data_quality_ok: bool = True, execution_cost_degraded: bool = False) -> SurvivalState:
        if not data_quality_ok or drawdown >= self.halt_drawdown or daily_loss >= self.halt_drawdown or consecutive_losses >= self.max_consecutive_losses:
            self.state = SurvivalState.HALTED
        elif execution_cost_degraded or drawdown >= self.defensive_drawdown:
            self.state = SurvivalState.DEFENSIVE
        elif drawdown >= self.caution_drawdown:
            self.state = SurvivalState.CAUTION
        elif self.state != SurvivalState.HALTED:
            self.state = SurvivalState.NORMAL
        return self.state

    def allows_new_trade(self) -> bool:
        return self.state in {SurvivalState.NORMAL, SurvivalState.CAUTION}
