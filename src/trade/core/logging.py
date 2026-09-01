"""Structured logging configuration using structlog.

Two output modes:
  - "dev"  → colorized, human-readable console output
  - "json" → machine-parsable JSON lines for production
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(level: str = "INFO", fmt: str = "dev") -> None:
    """Configure structlog and stdlib logging.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: Output format — "dev" for pretty console, "json" for JSON lines.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors applied to every log entry
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if fmt == "json":
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Quieten noisy third-party loggers
    for noisy in ("urllib3", "httpx", "yfinance", "mlflow", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structlog logger."""
    return structlog.get_logger(name)
