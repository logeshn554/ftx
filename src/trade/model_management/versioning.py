"""Semantic versioning for model checkpoints."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from trade.core.types import ModelVersion, ModelStage

logger = logging.getLogger(__name__)


class VersionManager:
    """Manages semantic versioning of model checkpoints.

    Version scheme: major.minor.patch
        - major: Architecture change (e.g. MLP → LSTM)
        - minor: Retraining with same architecture (new data/hyperparams)
        - patch: Bug fix or config tweak
    """

    def __init__(self) -> None:
        self._current: ModelVersion | None = None

    def set_current(self, version: ModelVersion) -> None:
        """Set the current version."""
        self._current = version
        logger.info("Current model version: %s", version.tag)

    def next_minor(self, **kwargs: Any) -> ModelVersion:
        """Create next minor version (retraining)."""
        if self._current is None:
            return ModelVersion(major=0, minor=1, patch=0, **kwargs)
        return ModelVersion(
            major=self._current.major,
            minor=self._current.minor + 1,
            patch=0,
            stage=ModelStage.CANDIDATE,
            **kwargs,
        )

    def next_major(self, **kwargs: Any) -> ModelVersion:
        """Create next major version (architecture change)."""
        if self._current is None:
            return ModelVersion(major=1, minor=0, patch=0, **kwargs)
        return ModelVersion(
            major=self._current.major + 1,
            minor=0,
            patch=0,
            stage=ModelStage.CANDIDATE,
            **kwargs,
        )

    def next_patch(self, **kwargs: Any) -> ModelVersion:
        """Create next patch version (bug fix)."""
        if self._current is None:
            return ModelVersion(major=0, minor=0, patch=1, **kwargs)
        return ModelVersion(
            major=self._current.major,
            minor=self._current.minor,
            patch=self._current.patch + 1,
            stage=ModelStage.CANDIDATE,
            **kwargs,
        )

    @property
    def current(self) -> ModelVersion | None:
        return self._current

    @staticmethod
    def compute_dataset_hash(data_config: dict) -> str:
        """Compute a hash of the dataset configuration for reproducibility."""
        raw = json.dumps(data_config, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
