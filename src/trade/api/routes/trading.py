"""Trading control endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request, HTTPException

from trade.agent.inference import ProductionAgent

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/start")
async def start_trading(request: Request):
    """Start the production trading agent."""
    try:
        config = request.app.state.config
        
        # Load agent if not already loaded
        if not hasattr(request.app.state, "production_agent") or request.app.state.production_agent is None:
            logger.info("Loading production agent from %s", config.model.champion_path)
            agent = ProductionAgent(model_path=config.model.champion_path)
            request.app.state.production_agent = agent
        else:
            agent = request.app.state.production_agent

        # Enable risk engine
        if hasattr(request.app.state, "risk_engine"):
            request.app.state.risk_engine.enable_trading()
            logger.info("Risk engine enabled")

        return {
            "status": "started",
            "message": "Production agent started",
            "model_version": agent.model_version if hasattr(agent, "model_version") else "unknown",
        }
    except Exception as e:
        logger.error("Failed to start trading: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_trading(request: Request):
    """Gracefully stop the trading agent."""
    try:
        if hasattr(request.app.state, "risk_engine"):
            request.app.state.risk_engine.disable_trading()
            logger.info("Risk engine disabled")

        if hasattr(request.app.state, "production_agent"):
            request.app.state.production_agent = None

        return {
            "status": "stopped",
            "message": "Production agent stopped",
        }
    except Exception as e:
        logger.error("Failed to stop trading: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def trading_status(request: Request):
    """Get current trading status."""
    try:
        agent_loaded = (
            hasattr(request.app.state, "production_agent") and
            request.app.state.production_agent is not None
        )
        
        risk_engine = getattr(request.app.state, "risk_engine", None)
        broker = getattr(request.app.state, "broker", None)
        
        trading_enabled = risk_engine.is_trading_enabled if risk_engine else False
        positions = list(broker.get_positions().keys()) if broker else []
        daily_pnl = risk_engine._daily_pnl if risk_engine else 0.0
        
        return {
            "status": "active" if trading_enabled else "idle",
            "agent_loaded": agent_loaded,
            "model_version": getattr(request.app.state.production_agent, "model_version", None) if agent_loaded else None,
            "trading_enabled": trading_enabled,
            "positions": positions,
            "open_position_count": len(positions),
            "daily_pnl": daily_pnl,
            "total_return": 0.0,
        }
    except Exception as e:
        logger.error("Failed to get trading status: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
