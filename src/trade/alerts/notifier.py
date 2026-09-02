"""Alert notifications for critical trading events.

Supports Telegram and email alerts for circuit breaker trips and performance issues.
Configured via environment variables.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import threading
from email.message import EmailMessage

import requests

from trade.core.secrets import get_alert_config

logger = logging.getLogger(__name__)


class AlertNotifier:
    """Send alerts via Telegram and/or email."""

    def __init__(self) -> None:
        self.config = get_alert_config()
        self._request_timeout = 5

    def send(self, subject: str, body: str, severity: str = "WARNING") -> None:
        """Send alert to all configured channels.

        Args:
            subject: Alert subject line.
            body: Alert message body.
            severity: One of CRITICAL, WARNING, INFO.
        """
        message = f"🔴 [{severity}] {subject}\n\n{body}"

        # Send in background threads to avoid blocking
        if self.config.telegram_enabled:
            threading.Thread(
                target=self._send_telegram,
                args=(message,),
                daemon=True,
            ).start()

        if self.config.email_enabled:
            threading.Thread(
                target=self._send_email,
                args=(subject, body, severity),
                daemon=True,
            ).start()

    def _send_telegram(self, message: str) -> None:
        """Send message via Telegram bot."""
        if not self.config.telegram_enabled:
            return

        try:
            url = f"https://api.telegram.org/bot{self.config.telegram_token.get_secret_value()}/sendMessage"
            payload = {
                "chat_id": self.config.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }
            response = requests.post(url, json=payload, timeout=self._request_timeout)
            response.raise_for_status()
            logger.debug("Telegram alert sent")
        except Exception as e:
            logger.error("Failed to send Telegram alert: %s", e)

    def _send_email(self, subject: str, body: str, severity: str) -> None:
        """Send alert via email."""
        if not self.config.email_enabled:
            return

        try:
            msg = EmailMessage()
            msg["Subject"] = f"[{severity}] TRADING ALERT: {subject}"
            msg["From"] = self.config.alert_email
            msg["To"] = self.config.alert_email
            msg.set_content(body)

            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port, timeout=self._request_timeout) as smtp:
                # Use TLS for secure connection
                smtp.starttls()
                smtp.send_message(msg)

            logger.debug("Email alert sent")
        except Exception as e:
            logger.error("Failed to send email alert: %s", e)


# Global notifier instance
_notifier: AlertNotifier | None = None


def get_notifier() -> AlertNotifier:
    """Get or create the global alert notifier."""
    global _notifier
    if _notifier is None:
        _notifier = AlertNotifier()
    return _notifier


def on_circuit_breaker_tripped(event: dict) -> None:
    """Handle circuit breaker tripped event."""
    notifier = get_notifier()
    notifier.send(
        subject="Circuit Breaker TRIPPED",
        body=f"Reason: {event.get('reason', 'Unknown')}\n"
              f"Cooldown: {event.get('cooldown_seconds', 3600):.0f}s\n"
              f"Time: {event.get('timestamp', 'Unknown')}",
        severity="CRITICAL",
    )


def on_performance_degraded(event: dict) -> None:
    """Handle performance degradation event."""
    notifier = get_notifier()
    notifier.send(
        subject=f"Performance Degraded: {event.get('metric_name', 'Unknown')}",
        body=f"Current Value: {event.get('current_value', 'N/A'):.4f}\n"
              f"Threshold: {event.get('threshold', 'N/A'):.4f}\n"
              f"Time: {event.get('timestamp', 'Unknown')}",
        severity="WARNING",
    )


def on_trading_disabled(event: dict) -> None:
    """Handle trading disabled event."""
    notifier = get_notifier()
    notifier.send(
        subject="Trading Disabled",
        body=f"Reason: {event.get('reason', 'Manual disable')}\n"
              f"Time: {event.get('timestamp', 'Unknown')}",
        severity="WARNING",
    )
