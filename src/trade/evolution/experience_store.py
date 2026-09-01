"""Immutable trade experience store for evolution and analysis."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Iterator

from trade.experience.schema import TradeExperience


class ImmutableExperienceStore:
    """Append-only store; records are never modified after write."""

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else None
        self._records: list[TradeExperience] = []
        self._lock = RLock()
        if self._path and self._path.exists():
            self._load()

    def append(self, record: TradeExperience) -> None:
        with self._lock:
            self._records.append(record)
            if self._path:
                self._persist()

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[TradeExperience]:
        with self._lock:
            yield from list(self._records)

    def query(self, strategy: str | None = None, regime: str | None = None) -> list[TradeExperience]:
        with self._lock:
            out = list(self._records)
        if strategy:
            out = [r for r in out if r.strategy == strategy]
        if regime:
            out = [r for r in out if r.regime == regime]
        return out

    def aggregate_by_strategy_regime(self) -> dict[str, dict[str, dict]]:
        stats: dict[str, dict[str, dict]] = {}
        for r in self:
            stats.setdefault(r.strategy, {}).setdefault(r.regime, {"sample_count": 0, "net_pnl": 0.0, "wins": 0})
            bucket = stats[r.strategy][r.regime]
            bucket["sample_count"] += 1
            bucket["net_pnl"] += r.net_pnl
            if r.net_pnl > 0:
                bucket["wins"] += 1
        for strat in stats:
            for regime in stats[strat]:
                b = stats[strat][regime]
                n = b["sample_count"]
                b["expectancy"] = b["net_pnl"] / n if n else 0.0
                b["win_rate"] = b["wins"] / n if n else 0.0
        return stats

    def _persist(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(r) for r in self._records]
        for item in payload:
            if hasattr(item.get("timestamp"), "isoformat"):
                item["timestamp"] = item["timestamp"].isoformat()
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        import datetime as dt

        raw = json.loads(self._path.read_text(encoding="utf-8"))
        for item in raw:
            item["timestamp"] = dt.datetime.fromisoformat(item["timestamp"])
            item["market_features"] = tuple(tuple(x) for x in item["market_features"])
            self._records.append(TradeExperience(**item))
