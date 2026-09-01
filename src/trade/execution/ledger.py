"""Canonical cash + position ledger with equity reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from trade.execution.accounting import ClosedTrade, PositionAccounting
from trade.execution.cost_model import CostConfig, CostModel
from trade.execution.position import Position, Side


@dataclass(frozen=True)
class LedgerSnapshot:
    cash: float
    equity: float
    position_value: float
    realized_pnl: float
    unrealized_pnl: float
    total_fees: float
    total_slippage: float
    turnover: float


class Ledger:
    """Single-source accounting for paper and research execution."""

    def __init__(self, initial_cash: float, cost_config: CostConfig | None = None):
        cfg = cost_config or CostConfig()
        self.cost_model = CostModel(cfg)
        self._book = PositionAccounting(
            initial_cash,
            fee_pct=self.cost_model.fee_rate,
            slippage_pct=cfg.entry_slippage,
        )
        self.initial_cash = float(initial_cash)
        self._position_meta: Position | None = None
        self._closed: list[ClosedTrade] = []

    @property
    def cash(self) -> float:
        return self._book.cash

    @property
    def open_position(self) -> Position | None:
        return self._position_meta

    def equity(self, market_price: float) -> float:
        return self._book.equity(market_price)

    def snapshot(self, market_price: float) -> LedgerSnapshot:
        pos_val = self._book.position * market_price
        return LedgerSnapshot(
            cash=self._book.cash,
            equity=self.equity(market_price),
            position_value=pos_val,
            realized_pnl=self._book.realized_pnl,
            unrealized_pnl=self._book.unrealized_pnl(market_price),
            total_fees=self._book.total_fees,
            total_slippage=self._book.total_slippage_cost,
            turnover=self._book.turnover,
        )

    def reconcile(self, market_price: float, tol: float = 1e-6) -> bool:
        """equity = cash + marked position value (long: +qty*price, short: liability)."""
        snap = self.snapshot(market_price)
        expected = snap.cash + self._book.position * market_price
        return abs(snap.equity - expected) <= tol

    def enter_position(self, symbol: str, side: Side, quantity: float, market_price: float) -> bool:
        if self._book.position:
            return False
        if not self._book.open(side, quantity, market_price):
            return False
        now = datetime.now(timezone.utc)
        fill = self._book.entry_price
        slip = abs(fill - market_price) * quantity
        self._position_meta = Position(
            symbol=symbol,
            side=side,
            entry_price=fill,
            entry_time=now,
            quantity=quantity,
            entry_fee=self._book.entry_fee,
            current_price=market_price,
            slippage_cost=slip,
        )
        return True

    def close_position(self, market_price: float) -> ClosedTrade | None:
        if not self._book.position or self._position_meta is None:
            return None
        trade = self._book.close(market_price)
        if trade is None:
            return None
        now = datetime.now(timezone.utc)
        self._position_meta.close(
            exit_price=trade.exit_price,
            exit_fee=trade.exit_fee,
            slippage_cost=trade.slippage_cost,
            exit_time=now,
        )
        self._position_meta.duration = trade.duration
        self._closed.append(trade)
        self._position_meta = None
        return trade

    def reverse(self, symbol: str, new_side: Side, quantity: float, market_price: float) -> bool:
        if self._book.position:
            self.close_position(market_price)
        return self.enter_position(symbol, new_side, quantity, market_price)

    def advance_bar(self) -> None:
        self._book.advance()
        if self._position_meta is not None:
            self._position_meta.duration += 1

    def round_trip_cost_pct(self) -> float:
        return self.cost_model.estimated_round_trip_cost_pct()
