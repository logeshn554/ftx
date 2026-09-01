"""Trading control endpoints."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.post("/start")
async def start_trading():
    """Start the production trading agent."""
    return {"status": "started", "message": "Production agent started"}


@router.post("/stop")
async def stop_trading():
    """Gracefully stop the trading agent."""
    return {"status": "stopped", "message": "Production agent stopped"}


@router.get("/status")
async def trading_status():
    """Get current trading status."""
    return {
        "status": "idle",
        "agent_loaded": False,
        "model_version": None,
        "positions": [],
        "daily_pnl": 0.0,
        "total_return": 0.0,
    }
