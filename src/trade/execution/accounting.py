"""Deterministic, price-based accounting for a single tradable position."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class ClosedTrade:
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    entry_fee: float
    exit_fee: float
    slippage_cost: float
    gross_pnl: float
    net_pnl: float
    return_pct: float
    duration: int

    def to_dict(self) -> dict:
        return asdict(self)


class PositionAccounting:
    """Cash, position, fills, and PnL with no dependency on dataframe labels.

    ``market_price`` is always the currently observable price.  Entry and exit
    fills are adjusted adversely for slippage, while unrealized PnL is valued
    at the unadjusted current market price.
    """

    def __init__(self, initial_cash: float, fee_pct: float = 0.001, slippage_pct: float = 0.0005):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_pct = float(fee_pct)
        self.slippage_pct = float(slippage_pct)
        self.position = 0.0  # signed: positive long, negative short
        self.entry_price = 0.0
        self.entry_fee = 0.0
        self.entry_slippage_cost = 0.0
        self.duration = 0
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.total_slippage_cost = 0.0
        self.turnover = 0.0

    @property
    def side(self) -> Side | None:
        return "BUY" if self.position > 0 else "SELL" if self.position < 0 else None

    @property
    def quantity(self) -> float:
        return abs(self.position)

    def equity(self, market_price: float) -> float:
        return self.cash + self.position * float(market_price)

    def unrealized_pnl(self, market_price: float) -> float:
        if not self.position:
            return 0.0
        direction = 1.0 if self.position > 0 else -1.0
        return direction * (float(market_price) - self.entry_price) * self.quantity - self.entry_fee

    def open(self, side: Side, quantity: float, market_price: float) -> bool:
        if quantity <= 0 or self.position:
            return False
        direction = 1.0 if side == "BUY" else -1.0
        fill = float(market_price) * (1 + self.slippage_pct if side == "BUY" else 1 - self.slippage_pct)
        notional = quantity * fill
        fee = notional * self.fee_pct
        # Longs require funding; shorts are collateralized by their sale proceeds.
        if side == "BUY" and self.cash + 1e-12 < notional + fee:
            return False
        self.position = direction * quantity
        self.entry_price = fill
        self.entry_fee = fee
        self.entry_slippage_cost = abs(fill - float(market_price)) * quantity
        self.total_fees += fee
        self.total_slippage_cost += self.entry_slippage_cost
        self.turnover += notional
        self.cash += -notional - fee if side == "BUY" else notional - fee
        self.duration = 0
        return True

    def close(self, market_price: float) -> ClosedTrade | None:
        if not self.position:
            return None
        side = self.side
        assert side is not None
        qty = self.quantity
        fill = float(market_price) * (1 - self.slippage_pct if side == "BUY" else 1 + self.slippage_pct)
        notional = qty * fill
        exit_fee = notional * self.fee_pct
        direction = 1.0 if side == "BUY" else -1.0
        gross = direction * (fill - self.entry_price) * qty
        net = gross - self.entry_fee - exit_fee
        exit_slippage = abs(fill - float(market_price)) * qty
        basis = self.entry_price * qty
        trade = ClosedTrade(side, self.entry_price, fill, qty, self.entry_fee, exit_fee,
                            self.entry_slippage_cost + exit_slippage, gross, net,
                            100 * net / basis if basis else 0.0, self.duration)
        self.cash += notional - exit_fee if side == "BUY" else -notional - exit_fee
        self.realized_pnl += net
        self.total_fees += exit_fee
        self.total_slippage_cost += exit_slippage
        self.turnover += notional
        self.position = self.entry_price = self.entry_fee = self.entry_slippage_cost = 0.0
        self.duration = 0
        return trade

    def advance(self) -> None:
        if self.position:
            self.duration += 1
