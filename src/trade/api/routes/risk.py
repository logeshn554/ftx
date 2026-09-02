"""Risk management REST endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from trade.risk.limits import RiskLimits

logger = logging.getLogger(__name__)

router = APIRouter()


class RiskLimitsUpdate(BaseModel):
    """Request body for updating risk limits."""

    max_position_pct: float | None = None
    max_daily_loss_pct: float | None = None
    max_weekly_loss_pct: float | None = None
    max_total_drawdown_pct: float | None = None
    max_order_pct: float | None = None
    max_leverage: float | None = None


@router.get("/status")
async def risk_status(request: Request):
    """Get current risk engine status."""
    try:
        risk_engine = getattr(request.app.state, "risk_engine", None)
        circuit_breaker = getattr(request.app.state, "circuit_breaker", None)
        broker = getattr(request.app.state, "broker", None)

        if not risk_engine:
            return {"error": "Risk engine not initialized"}

        positions = broker.get_positions() if broker else {}
        daily_pnl = risk_engine._daily_pnl if risk_engine else 0.0

        return {
            "trading_enabled": risk_engine.is_trading_enabled,
            "circuit_breaker_state": circuit_breaker.state.value if circuit_breaker else "UNKNOWN",
            "daily_loss_pct": daily_pnl / (risk_engine._daily_start_equity or 1) * 100 if risk_engine._daily_start_equity else 0.0,
            "open_positions": len(positions),
            "orders_today": risk_engine._daily_order_count,
            "limits": {
                "max_position_pct": risk_engine.limits.max_position_pct,
                "max_daily_loss_pct": risk_engine.limits.max_daily_loss_pct,
                "max_leverage": risk_engine.limits.max_leverage,
            },
        }
    except Exception as e:
        logger.error("Failed to get risk status: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/limits")
async def update_limits(request: Request, update: RiskLimitsUpdate):
    """Update risk limits."""
    try:
        risk_engine = getattr(request.app.state, "risk_engine", None)
        if not risk_engine:
            raise HTTPException(status_code=400, detail="Risk engine not initialized")

        limits = risk_engine.limits

        # Update only provided fields
        if update.max_position_pct is not None:
            limits.max_position_pct = update.max_position_pct
        if update.max_daily_loss_pct is not None:
            limits.max_daily_loss_pct = update.max_daily_loss_pct
        if update.max_weekly_loss_pct is not None:
            limits.max_weekly_loss_pct = update.max_weekly_loss_pct
        if update.max_total_drawdown_pct is not None:
            limits.max_total_drawdown_pct = update.max_total_drawdown_pct
        if update.max_order_pct is not None:
            limits.max_order_pct = update.max_order_pct
        if update.max_leverage is not None:
            limits.max_leverage = update.max_leverage

        logger.info("Risk limits updated: %s", update)

        return {
            "status": "updated",
            "limits": {
                "max_position_pct": limits.max_position_pct,
                "max_daily_loss_pct": limits.max_daily_loss_pct,
                "max_leverage": limits.max_leverage,
            },
        }
    except Exception as e:
        logger.error("Failed to update risk limits: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/circuit-breaker/reset")
async def reset_circuit_breaker(request: Request):
    """Manually reset the circuit breaker."""
    try:
        circuit_breaker = getattr(request.app.state, "circuit_breaker", None)
        if not circuit_breaker:
            raise HTTPException(status_code=400, detail="Circuit breaker not initialized")

        # Circuit breaker can only be reset from OPEN state
        from trade.core.types import CircuitState

        if circuit_breaker.state == CircuitState.OPEN:
            circuit_breaker.reset()
            logger.info("Circuit breaker manually reset")
            return {
                "status": "reset",
                "circuit_breaker_state": circuit_breaker.state.value,
            }
        else:
            return {
                "status": "no_action",
                "message": f"Circuit breaker is {circuit_breaker.state.value}, cannot reset",
                "circuit_breaker_state": circuit_breaker.state.value,
            }
    except Exception as e:
        logger.error("Failed to reset circuit breaker: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
