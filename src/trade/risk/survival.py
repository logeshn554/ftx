"""Portfolio survival state machine; HALTED always blocks new entries."""
from __future__ import annotations
from enum import Enum


class SurvivalState(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    DEFENSIVE = "DEFENSIVE"
    HALTED = "HALTED"


class SurvivalController:
    """Multi-tiered capital defense and survival controller.

    States:
        NORMAL: Standard trading with full position limits.
        CAUTION: Drawdown elevated; reduce risk budget.
        DEFENSIVE: Severe drawdown, drift, or cost degradation; tight limits.
        HALTED: Critical loss, daily loss breach, data failure, or model degradation;
                blocks all new trade entries.
    """

    def __init__(
        self,
        caution_drawdown: float = 0.05,
        defensive_drawdown: float = 0.10,
        halt_drawdown: float = 0.20,
        max_daily_loss: float = 0.05,
        max_consecutive_losses: int = 5,
    ):
        self.caution_drawdown = caution_drawdown
        self.defensive_drawdown = defensive_drawdown
        self.halt_drawdown = halt_drawdown
        self.max_daily_loss = max_daily_loss
        self.max_consecutive_losses = max_consecutive_losses
        self.state = SurvivalState.NORMAL
        self.halt_reason: str | None = None

    def update(
        self,
        drawdown: float,
        consecutive_losses: int = 0,
        daily_loss: float = 0.0,
        data_quality_ok: bool = True,
        execution_cost_degraded: bool = False,
        model_degradation: bool = False,
        drift_detected: bool = False,
    ) -> SurvivalState:
        """Evaluate market and account health metrics to update survival state."""
        # Hard halt conditions
        if not data_quality_ok:
            self.state = SurvivalState.HALTED
            self.halt_reason = "DATA_QUALITY_FAILURE"
        elif drawdown >= self.halt_drawdown:
            self.state = SurvivalState.HALTED
            self.halt_reason = f"MAX_DRAWDOWN_BREACH ({drawdown:.2%})"
        elif daily_loss >= self.max_daily_loss:
            self.state = SurvivalState.HALTED
            self.halt_reason = f"DAILY_LOSS_BREACH ({daily_loss:.2%})"
        elif consecutive_losses >= self.max_consecutive_losses:
            self.state = SurvivalState.HALTED
            self.halt_reason = f"CONSECUTIVE_LOSS_LIMIT ({consecutive_losses})"
        elif model_degradation:
            self.state = SurvivalState.HALTED
            self.halt_reason = "MODEL_DEGRADATION_DETECTED"
        # Defensive conditions
        elif drift_detected or execution_cost_degraded or drawdown >= self.defensive_drawdown:
            self.state = SurvivalState.DEFENSIVE
            self.halt_reason = None
        # Caution conditions
        elif drawdown >= self.caution_drawdown:
            self.state = SurvivalState.CAUTION
            self.halt_reason = None
        # Normal (only if not currently halted without manual/cooldown recovery)
        elif self.state != SurvivalState.HALTED:
            self.state = SurvivalState.NORMAL
            self.halt_reason = None

        return self.state

    def recover(self, force: bool = False) -> bool:
        """Attempt recovery from HALTED state."""
        if self.state == SurvivalState.HALTED or force:
            self.state = SurvivalState.DEFENSIVE  # Step down to DEFENSIVE first for safety
            self.halt_reason = None
            return True
        return False

    def allows_new_trade(self) -> bool:
        """HALTED strictly disallows opening new trades."""
        return self.state in {SurvivalState.NORMAL, SurvivalState.CAUTION}
