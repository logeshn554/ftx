"""Dashboard server: serves the static frontend or launches uvicorn with FastAPI app."""

from __future__ import annotations

import os
from pathlib import Path
import uvicorn

from trade.api.app import create_app
from trade.core.config import AppConfig, build_config
from trade.core.logging import get_logger

logger = get_logger("dashboard.server")


def start_dashboard_server(
    host: str | None = None,
    port: int | None = None,
    config: AppConfig | None = None,
) -> None:
    """Start the dashboard and API server.

    Args:
        host: Host IP or hostname. Defaults to config value.
        port: Port number. Defaults to config value.
        config: Application configuration. Defaults to loaded config.
    """
    if config is None:
        config = build_config()

    server_host = host or config.api.host
    server_port = port or config.api.port

    logger.info("Starting trading dashboard server", host=server_host, port=server_port)
    app = create_app(config=config)
    uvicorn.run(app, host=server_host, port=server_port, log_level="info")


if __name__ == "__main__":
    start_dashboard_server()
