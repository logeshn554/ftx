"""Secrets management with .env file loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import SecretStr, BaseModel


class BrokerConfig(BaseModel):
    """Broker credentials with secret masking."""

    api_key: SecretStr = SecretStr("")
    api_secret: SecretStr = SecretStr("")
    base_url: str = "https://api.binance.com"

    @property
    def has_credentials(self) -> bool:
        """Check if both API key and secret are present."""
        return bool(
            self.api_key.get_secret_value() and
            self.api_secret.get_secret_value()
        )

    def validate_for_live_trading(self) -> list[str]:
        """Validate credentials are safe for live trading."""
        errors = []
        key_val = self.api_key.get_secret_value()
        secret_val = self.api_secret.get_secret_value()

        if not key_val:
            errors.append("API key must not be empty for live trading")
        if not secret_val:
            errors.append("API secret must not be empty for live trading")
        if len(key_val) < 10:
            errors.append("API key appears too short (likely invalid)")
        if len(secret_val) < 10:
            errors.append("API secret appears too short (likely invalid)")

        return errors


class AlertConfig(BaseModel):
    """Alert notification credentials."""

    telegram_token: SecretStr = SecretStr("")
    telegram_chat_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    alert_email: str = ""

    @property
    def telegram_enabled(self) -> bool:
        """Check if Telegram alerts are configured."""
        return bool(
            self.telegram_token.get_secret_value() and
            self.telegram_chat_id
        )

    @property
    def email_enabled(self) -> bool:
        """Check if email alerts are configured."""
        return bool(self.smtp_host and self.alert_email)


def load_env() -> None:
    """Load environment variables from .env file if it exists."""
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(str(env_file))


def get_broker_config() -> BrokerConfig:
    """Load broker configuration from environment variables."""
    load_env()
    return BrokerConfig(
        api_key=SecretStr(os.environ.get("TRADE_BROKER_API_KEY", "")),
        api_secret=SecretStr(os.environ.get("TRADE_BROKER_API_SECRET", "")),
        base_url=os.environ.get("TRADE_BROKER_BASE_URL", "https://api.binance.com"),
    )


def get_alert_config() -> AlertConfig:
    """Load alert configuration from environment variables."""
    load_env()
    return AlertConfig(
        telegram_token=SecretStr(os.environ.get("TRADE_TELEGRAM_TOKEN", "")),
        telegram_chat_id=os.environ.get("TRADE_TELEGRAM_CHAT_ID", ""),
        smtp_host=os.environ.get("TRADE_SMTP_HOST", ""),
        smtp_port=int(os.environ.get("TRADE_SMTP_PORT", "587")),
        alert_email=os.environ.get("TRADE_ALERT_EMAIL", ""),
    )


def get_api_key() -> str:
    """Get the API key from environment."""
    load_env()
    return os.environ.get("TRADE_API_KEY", "")


def get_ws_token() -> str:
    """Get the WebSocket token from environment."""
    load_env()
    return os.environ.get("TRADE_WS_TOKEN", "")
