"""FastAPI application factory with lifespan management."""

from __future__ import annotations

import logging
import signal
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from trade.api.middleware import APIKeyMiddleware
from trade.api.routes import trading, models, risk, ws
from trade.core.config import AppConfig, build_config
from trade.core.logging import setup_logging
from trade.core.types import CircuitState

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown with component initialization."""
    config: AppConfig = app.state.config
    setup_logging(level=config.log_level, fmt=config.log_format)
    logger.info("Trading system startup...")

    # Initialize components (risk engine, broker, etc.) can be done here
    # For now, this is a placeholder for component initialization

    yield

    # FIX 19: Graceful shutdown
    logger.info("Trading system shutdown initiated...")

    # Disable trading on shutdown
    risk_engine = getattr(app.state, "risk_engine", None)
    if risk_engine:
        risk_engine.disable_trading()
        logger.info("Trading disabled on shutdown")

    # Close any open positions
    broker = getattr(app.state, "broker", None)
    if broker:
        positions = broker.get_positions()
        if positions:
            logger.warning(
                "Shutdown with %d open positions (not auto-closing)",
                len(positions),
            )

    logger.info("System shutdown complete")


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Application configuration. If None, loads from defaults.

    Returns:
        Configured FastAPI app.
    """
    if config is None:
        config = build_config()

    app = FastAPI(
        title="Self-Evolving Trading System",
        description="Production RL trading agent with safety-first architecture",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store config in app state
    app.state.config = config

    # FIX 15: Fix CORS configuration (no wildcard)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,  # Explicit origins
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT"],  # Explicit methods
        allow_headers=["Content-Type", "X-API-Key"],  # Explicit headers
    )

    # FIX 12: Add API authentication middleware
    app.add_middleware(APIKeyMiddleware)

    # Include route modules
    app.include_router(trading.router, prefix="/trading", tags=["Trading"])
    app.include_router(models.router, prefix="/models", tags=["Models"])
    app.include_router(risk.router, prefix="/risk", tags=["Risk"])
    app.include_router(ws.router, tags=["WebSocket"])

    # Serve dashboard static files
    dashboard_dir = Path(__file__).parent.parent.parent.parent / "dashboard"
    if dashboard_dir.exists():
        app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

    # FIX 26: Real health check that tests components
    @app.get("/health")
    async def health(request: Request):
        """Health check endpoint that validates system components."""
        checks = {}
        overall_status = "ok"

        # Check model loaded
        agent = getattr(request.app.state, "production_agent", None)
        checks["model_loaded"] = agent is not None

        # Check circuit breaker
        cb = getattr(request.app.state, "circuit_breaker", None)
        if cb:
            checks["circuit_breaker_state"] = cb.state.value
            if cb.state == CircuitState.OPEN:
                overall_status = "degraded"

        # Check risk engine
        re = getattr(request.app.state, "risk_engine", None)
        if re:
            checks["trading_enabled"] = re.is_trading_enabled
            checks["daily_reset_initialized"] = re._daily_start_equity is not None
            if not checks.get("daily_reset_initialized"):
                overall_status = "degraded"

        if not checks.get("model_loaded"):
            overall_status = "degraded"

        status_code = 200 if overall_status == "ok" else 503
        return JSONResponse(
            {
                "status": overall_status,
                "version": "0.1.0",
                "checks": checks,
            },
            status_code=status_code,
        )

    return app

