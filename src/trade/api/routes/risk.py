"""Risk management REST endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
async def risk_status():
    """Get current risk engine status."""
    return {
        "trading_enabled": True,
        "circuit_breaker": "CLOSED",
        "daily_loss_pct": 0.0,
        "open_positions": 0,
        "orders_today": 0,
    }


@router.put("/limits")
async def update_limits(limits: dict):
    """Update risk limits."""
    return {"status": "updated", "limits": limits}


@router.post("/circuit-breaker/reset")
async def reset_circuit_breaker():
    """Manually reset the circuit breaker."""
    return {"status": "reset", "circuit_breaker": "CLOSED"}
