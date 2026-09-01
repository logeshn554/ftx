"""CLI: Start the production trading agent."""

import argparse
import sys

import uvicorn

from trade.core.config import build_config
from trade.core.logging import setup_logging, get_logger
from trade.api.app import create_app


def main():
    parser = argparse.ArgumentParser(description="Start the trading system")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--mode", choices=["paper", "shadow", "live"], default=None)
    args = parser.parse_args()

    config = build_config(args.config)
    if args.host:
        config.api.host = args.host
    if args.port:
        config.api.port = args.port
    if args.mode:
        config.trading.mode = args.mode

    setup_logging(config.log_level, config.log_format)
    log = get_logger("trade")

    log.info(
        "Starting trading system",
        mode=config.trading.mode,
        host=config.api.host,
        port=config.api.port,
    )

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level="info",
    )


if __name__ == "__main__":
    sys.exit(main())
