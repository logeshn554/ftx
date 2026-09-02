"""Multiple testing correction and trial ledger for statistical research integrity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import datetime as dt
import json
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from trade.validation import metrics as m


@dataclass(frozen=True)
class TrialRecord:
    trial_id: str
    timestamp: str
    model_version: str
    candidate_id: str
    parameters: dict[str, Any]
    oos_sharpe: float
    oos_return: float
    track_record_length: int


class ResearchTrialLedger:
    """Persistent ledger tracking every historical research trial for multiple-testing deflation."""

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path else None
        self._trials: list[TrialRecord] = []
        self._lock = RLock()
        if self._path and self._path.exists():
            self._load()

    def record_trial(
        self,
        trial_id: str,
        model_version: str,
        candidate_id: str,
        parameters: dict[str, Any],
        oos_sharpe: float,
        oos_return: float,
        track_record_length: int,
    ) -> TrialRecord:
        record = TrialRecord(
            trial_id=trial_id,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            model_version=model_version,
            candidate_id=candidate_id,
            parameters=parameters,
            oos_sharpe=float(oos_sharpe),
            oos_return=float(oos_return),
            track_record_length=int(track_record_length),
        )
        with self._lock:
            self._trials.append(record)
            if self._path:
                self._persist()
        return record

    def total_trials_count(self) -> int:
        with self._lock:
            return len(self._trials)

    def evaluate_candidate_dsr(
        self,
        estimated_sharpe: float,
        track_record_length: int,
        confidence_level: float = 0.90,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Compute Deflated Sharpe Ratio against all historical candidate trials."""
        with self._lock:
            trials_count = max(1, len(self._trials))
            sharpes = [t.oos_sharpe for t in self._trials]

        skewness = 0.0
        kurtosis = 3.0
        if len(sharpes) >= 5:
            r_arr = np.array(sharpes)
            mean_s = np.mean(r_arr)
            std_s = np.std(r_arr)
            if std_s > 1e-6:
                skewness = float(np.mean(((r_arr - mean_s) / std_s) ** 3))
                kurtosis = float(np.mean(((r_arr - mean_s) / std_s) ** 4))

        dsr_prob = m.deflated_sharpe_ratio(
            estimated_sharpe=estimated_sharpe,
            num_trials=trials_count,
            track_record_length=track_record_length,
            skewness=skewness,
            kurtosis=kurtosis,
        )

        passed = dsr_prob >= confidence_level
        audit = {
            "num_trials": trials_count,
            "estimated_sharpe": estimated_sharpe,
            "track_record_length": track_record_length,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "dsr_prob": dsr_prob,
            "confidence_level": confidence_level,
        }
        return passed, dsr_prob, audit

    def _persist(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in self._trials], f, indent=2)

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._trials = [TrialRecord(**d) for d in data]
        except Exception:
            self._trials = []
