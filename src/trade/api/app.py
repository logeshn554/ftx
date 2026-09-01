"""FastAPI application factory with lifespan management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from trade.api.routes import trading, models, risk, ws
from trade.core.config import AppConfig, build_config
from trade.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    config: AppConfig = app.state.config
    setup_logging(level=config.log_level, fmt=config.log_format)
    yield
    # Cleanup on shutdown


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

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include route modules
    app.include_router(trading.router, prefix="/trading", tags=["Trading"])
    app.include_router(models.router, prefix="/models", tags=["Models"])
    app.include_router(risk.router, prefix="/risk", tags=["Risk"])
    app.include_router(ws.router, tags=["WebSocket"])

    # Serve dashboard static files
    dashboard_dir = Path(__file__).parent.parent.parent.parent / "dashboard"
    if dashboard_dir.exists():
        app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": "0.1.0"}

    return app
