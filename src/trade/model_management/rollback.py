"""Rollback manager: safely revert to a previous model version."""

from __future__ import annotations

import logging

from trade.core.events import ModelRolledBack, event_bus
from trade.core.types import ModelStage
from trade.model_management.registry import ModelRegistry

logger = logging.getLogger(__name__)


class RollbackManager:
    """Manages safe rollback to previous model versions.

    Keeps a history of production model versions and enables
    one-command rollback with audit trail.
    """

    def __init__(self, registry: ModelRegistry, max_history: int = 5) -> None:
        self._registry = registry
        self._max_history = max_history
        self._rollback_history: list[dict] = []

    def rollback(self, reason: str = "") -> bool:
        """Rollback production to the most recent archived model.

        Args:
            reason: Human-readable reason for rollback.

        Returns:
            True if rollback was successful.
        """
        # Find the most recently archived model
        archived = self._registry.list_models(stage=ModelStage.ARCHIVED)
        if not archived:
            logger.error("No archived models available for rollback")
            return False

        # Sort by archived_at timestamp, most recent first
        archived.sort(
            key=lambda m: m.get("archived_at", ""),
            reverse=True,
        )

        target = archived[0]
        current_prod = self._registry.get_production_model()
        current_tag = current_prod["version"] if current_prod else "none"

        # Promote the archived model back to production
        success = self._registry.promote(
            target["version"],
            ModelStage.PRODUCTION,
            reason=f"Rollback: {reason}",
        )

        if success:
            self._rollback_history.append({
                "from_version": current_tag,
                "to_version": target["version"],
                "reason": reason,
            })

            # Trim history
            if len(self._rollback_history) > self._max_history:
                self._rollback_history = self._rollback_history[-self._max_history:]

            event_bus.publish_sync(
                ModelRolledBack(
                    from_version=current_tag,
                    to_version=target["version"],
                    reason=reason,
                )
            )

            logger.warning(
                "🔙 ROLLBACK: %s → %s | Reason: %s",
                current_tag,
                target["version"],
                reason,
            )

        return success

    def rollback_to(self, version_tag: str, reason: str = "") -> bool:
        """Rollback to a specific model version.

        Args:
            version_tag: The version tag to restore.
            reason: Reason for rollback.

        Returns:
            True if successful.
        """
        model = self._registry.get_model(version_tag)
        if model is None:
            logger.error("Model %s not found", version_tag)
            return False

        return self._registry.promote(
            version_tag,
            ModelStage.PRODUCTION,
            reason=f"Targeted rollback: {reason}",
        )

    @property
    def history(self) -> list[dict]:
        return self._rollback_history.copy()
