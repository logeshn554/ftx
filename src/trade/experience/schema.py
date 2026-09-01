"""Immutable, traceable trade experience records."""
from __future__ import annotations
from dataclasses import dataclass
import datetime as dt


@dataclass(frozen=True)
class TradeExperience:
    timestamp: dt.datetime
    symbol: str
    timeframe: str
    market_features: tuple[tuple[str, float], ...]
    regime: str
    regime_confidence: float
    strategy: str
    action: str
    signal_confidence: float
    expected_value: float
    entry_price: float
    exit_price: float | None
    quantity: float
    gross_pnl: float
    fees: float
    slippage: float
    net_pnl: float
    return_pct: float
    maximum_adverse_excursion: float
    maximum_favorable_excursion: float
    duration: int
    drawdown_before: float
    drawdown_after: float
    outcome: str
    model_version: str
    strategy_version: str
    feature_version: str

    def __post_init__(self):
        if self.quantity < 0 or self.entry_price <= 0:
            raise ValueError("invalid trade dimensions")
        if not self.model_version or not self.feature_version:
            raise ValueError("model and feature versions are required")

