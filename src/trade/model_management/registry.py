"""Model registry: track, version, and promote models.

Uses MLflow as the backing store for model tracking, with a clean
abstraction layer so the rest of the system doesn't depend on MLflow directly.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

from trade.core.events import ModelPromoted, event_bus
from trade.core.types import ModelStage, ModelVersion

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Manages model versions, stages, and promotion lifecycle.

    Models progress through stages:
        CANDIDATE → BACKTESTING → PAPER_TRADING → SHADOW → STAGING → PRODUCTION

    Only one model can be in PRODUCTION at a time.
    Previous production models are moved to ARCHIVED.
    """

    def __init__(self, registry_dir: str = "mlflow_registry") -> None:
        self._registry_dir = Path(registry_dir)
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, dict[str, Any]] = {}
        self._production_version: str | None = None

        # Load existing registry
        self._load_registry()

    def register(
        self,
        version: ModelVersion,
        checkpoint_path: str,
        metrics: dict[str, float] | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ) -> str:
        """Register a new model version.

        Args:
            version: The model version descriptor.
            checkpoint_path: Path to the saved model file.
            metrics: Training/eval metrics.
            hyperparameters: Training hyperparameters.

        Returns:
            The version tag string.
        """
        tag = version.tag

        self._models[tag] = {
            "version": tag,
            "major": version.major,
            "minor": version.minor,
            "patch": version.patch,
            "stage": version.stage.value,
            "checkpoint_path": checkpoint_path,
            "metrics": metrics or {},
            "hyperparameters": hyperparameters or {},
            "created_at": dt.datetime.utcnow().isoformat(),
            "promoted_at": None,
            "archived_at": None,
        }

        self._save_registry()
        logger.info("Registered model %s (stage: %s)", tag, version.stage.value)
        return tag

    def get_model(self, version_tag: str) -> dict[str, Any] | None:
        """Get model info by version tag."""
        return self._models.get(version_tag)

    def get_production_model(self) -> dict[str, Any] | None:
        """Get the current production model."""
        if self._production_version:
            return self._models.get(self._production_version)
        # Find by stage
        for model in self._models.values():
            if model["stage"] == ModelStage.PRODUCTION.value:
                self._production_version = model["version"]
                return model
        return None

    def list_models(
        self, stage: ModelStage | None = None
    ) -> list[dict[str, Any]]:
        """List all registered models, optionally filtered by stage."""
        models = list(self._models.values())
        if stage:
            models = [m for m in models if m["stage"] == stage.value]
        return sorted(models, key=lambda m: m["created_at"], reverse=True)

    def promote(
        self,
        version_tag: str,
        to_stage: ModelStage,
        reason: str = "",
    ) -> bool:
        """Promote a model to a new stage.

        If promoting to PRODUCTION, the current production model is archived.

        Args:
            version_tag: Version tag to promote.
            to_stage: Target stage.
            reason: Reason for promotion.

        Returns:
            True if promotion succeeded.
        """
        model = self._models.get(version_tag)
        if model is None:
            logger.error("Model %s not found in registry", version_tag)
            return False

        old_stage = model["stage"]

        # If promoting to production, archive the current production model
        if to_stage == ModelStage.PRODUCTION:
            current_prod = self.get_production_model()
            if current_prod and current_prod["version"] != version_tag:
                current_prod["stage"] = ModelStage.ARCHIVED.value
                current_prod["archived_at"] = dt.datetime.utcnow().isoformat()
                logger.info(
                    "Archived previous production model: %s",
                    current_prod["version"],
                )

            self._production_version = version_tag

            event_bus.publish_sync(
                ModelPromoted(
                    old_version=current_prod["version"] if current_prod else "none",
                    new_version=version_tag,
                    reason=reason,
                )
            )

        model["stage"] = to_stage.value
        model["promoted_at"] = dt.datetime.utcnow().isoformat()

        self._save_registry()
        logger.info(
            "Model %s promoted: %s → %s (%s)",
            version_tag, old_stage, to_stage.value, reason,
        )
        return True

    def reject(self, version_tag: str, reason: str = "") -> bool:
        """Reject a candidate model."""
        model = self._models.get(version_tag)
        if model is None:
            return False

        model["stage"] = ModelStage.REJECTED.value
        model["rejection_reason"] = reason
        self._save_registry()
        logger.info("Model %s REJECTED: %s", version_tag, reason)
        return True

    def get_checkpoint_path(self, version_tag: str) -> str | None:
        """Get the checkpoint file path for a model version."""
        model = self._models.get(version_tag)
        return model["checkpoint_path"] if model else None

    # -- Persistence -----------------------------------------------------------

    def _registry_path(self) -> Path:
        return self._registry_dir / "registry.json"

    def _save_registry(self) -> None:
        """Persist registry to disk."""
        data = {
            "production_version": self._production_version,
            "models": self._models,
        }
        with open(self._registry_path(), "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_registry(self) -> None:
        """Load registry from disk."""
        path = self._registry_path()
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self._models = data.get("models", {})
                self._production_version = data.get("production_version")
                logger.info(
                    "Loaded registry: %d models, production=%s",
                    len(self._models),
                    self._production_version,
                )
            except Exception:
                logger.warning("Failed to load registry, starting fresh")
                self._models = {}
