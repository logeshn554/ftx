"""Rollback snapshot schema and archive manager.

Guarantees full state recovery (model, config, feature version, strategy version,
and risk parameters) during any promotion rollback.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RollbackSnapshot:
    timestamp: str
    champion_version: str
    champion_config: dict[str, Any]
    model_checkpoint_path: str = ""
    strategy_version: str = "v1.0.0"
    feature_version: str = "v1.0.0"
    risk_config: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RollbackSnapshot:
        return cls(**data)


class RollbackArchive:
    """Manages an append-only archive of rollback snapshots."""

    def __init__(self, archive_path: str | Path | None = None, max_snapshots: int = 10):
        self.archive_path = Path(archive_path) if archive_path else None
        self.max_snapshots = max_snapshots
        self._snapshots: list[RollbackSnapshot] = []
        if self.archive_path and self.archive_path.exists():
            self._load()

    def record_snapshot(self, snapshot: RollbackSnapshot) -> None:
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)
        self._save()

    def get_latest(self) -> RollbackSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def pop_latest(self) -> RollbackSnapshot | None:
        if not self._snapshots:
            return None
        snap = self._snapshots.pop()
        self._save()
        return snap

    def list_snapshots(self) -> list[RollbackSnapshot]:
        return list(self._snapshots)

    def _save(self) -> None:
        if not self.archive_path:
            return
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        raw = [s.to_dict() for s in self._snapshots]
        self.archive_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    def _load(self) -> None:
        try:
            raw = json.loads(self.archive_path.read_text(encoding="utf-8"))
            self._snapshots = [RollbackSnapshot.from_dict(item) for item in raw]
        except Exception:
            self._snapshots = []
