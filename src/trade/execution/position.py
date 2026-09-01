"""Explicit position state for audit-grade execution accounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Side = Literal["BUY", "SELL"]


@dataclass
class Position:
    symbol: str
    side: Side
    entry_price: float
    entry_time: datetime
    quantity: float
    entry_fee: float
    current_price: float = 0.0
    exit_price: float | None = None
    exit_time: datetime | None = None
    exit_fee: float = 0.0
    slippage_cost: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0
    duration: int = 0

    @property
    def is_open(self) -> bool:
        return self.exit_price is None

    def mark(self, price: float) -> None:
        self.current_price = float(price)
        if self.is_open:
            direction = 1.0 if self.side == "BUY" else -1.0
            self.gross_pnl = direction * (self.current_price - self.entry_price) * self.quantity
            self.net_pnl = self.gross_pnl - self.entry_fee

    def close(self, exit_price: float, exit_fee: float, slippage_cost: float, exit_time: datetime) -> None:
        self.exit_price = float(exit_price)
        self.exit_time = exit_time
        self.exit_fee = float(exit_fee)
        self.slippage_cost = float(slippage_cost)
        direction = 1.0 if self.side == "BUY" else -1.0
        self.gross_pnl = direction * (self.exit_price - self.entry_price) * self.quantity
        self.net_pnl = self.gross_pnl - self.entry_fee - self.exit_fee - self.slippage_cost
        basis = self.entry_price * self.quantity
        self.return_pct = 100.0 * self.net_pnl / basis if basis else 0.0

    def unrealized_pnl(self) -> float:
        if not self.is_open:
            return self.net_pnl
        direction = 1.0 if self.side == "BUY" else -1.0
        gross = direction * (self.current_price - self.entry_price) * self.quantity
        return gross - self.entry_fee
