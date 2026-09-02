"""Configurable transaction cost model with canonical execution cost estimates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionCost:
    entry_fee: float
    exit_fee: float
    entry_slippage: float
    exit_slippage: float
    spread_cost: float
    total_cost: float

    @property
    def total_cost_fraction(self) -> float:
        return self.total_cost


def estimate_execution_cost(
    side: str,
    quantity: float,
    price: float,
    liquidity: float = 1.0,
    spread: float = 0.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> ExecutionCost:
    """Canonical execution cost calculator across all runtimes."""
    notional = float(quantity) * float(price)
    # Liquidity impact scaling (if liquidity < 1.0, slippage scales up)
    liq_multiplier = 1.0 / max(0.1, float(liquidity))
    effective_slippage = slippage_rate * liq_multiplier
    
    entry_fee = notional * fee_rate
    exit_fee = notional * fee_rate
    entry_slip = notional * effective_slippage
    exit_slip = notional * effective_slippage
    spread_cost = notional * (spread * 0.5)
    
    total = entry_fee + exit_fee + entry_slip + exit_slip + spread_cost
    return ExecutionCost(
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        entry_slippage=entry_slip,
        exit_slippage=exit_slip,
        spread_cost=spread_cost,
        total_cost=total,
    )


@dataclass(frozen=True)
class CostConfig:
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    entry_slippage: float = 0.0005
    exit_slippage: float = 0.0005
    spread_estimate: float = 0.0
    funding_rate: float = 0.0
    use_taker: bool = True


class CostModel:
    """Round-trip and per-leg cost estimates in return (fraction) units."""

    def __init__(self, config: CostConfig | None = None, stress: bool = False):
        self.config = config or CostConfig()
        self.fee_mult = 1.5 if stress else 1.0
        self.slippage_mult = 2.0 if stress else 1.0

    @property
    def fee_rate(self) -> float:
        return (self.config.taker_fee if self.config.use_taker else self.config.maker_fee) * self.fee_mult

    @property
    def slippage_rate(self) -> float:
        return self.config.entry_slippage * self.slippage_mult

    def estimate_cost(self, side: str = "BUY", quantity: float = 1.0, price: float = 1.0) -> ExecutionCost:
        return estimate_execution_cost(
            side=side,
            quantity=quantity,
            price=price,
            spread=self.config.spread_estimate,
            fee_rate=self.fee_rate,
            slippage_rate=self.slippage_rate,
        )

    def estimated_entry_cost(self, notional: float = 1.0) -> float:
        """Entry cost as fraction of notional (fee + slippage + half spread)."""
        fee = self.fee_rate
        slip = self.config.entry_slippage * self.slippage_mult
        spread = self.config.spread_estimate * 0.5
        return fee + slip + spread

    def estimated_exit_cost(self, notional: float = 1.0) -> float:
        fee = self.fee_rate
        slip = self.config.exit_slippage * self.slippage_mult
        spread = self.config.spread_estimate * 0.5
        return fee + slip + spread

    def estimated_round_trip_cost_pct(self) -> float:
        """Total round-trip cost as a percentage (e.g. 0.25 means 0.25%)."""
        return 100.0 * (self.estimated_entry_cost() + self.estimated_exit_cost())

    def estimated_round_trip_cost_fraction(self) -> float:
        return self.estimated_entry_cost() + self.estimated_exit_cost()

    def entry_fill_price(self, side: str, market_price: float) -> float:
        slip = self.config.entry_slippage * self.slippage_mult
        return market_price * (1 + slip) if side == "BUY" else market_price * (1 - slip)

    def exit_fill_price(self, side: str, market_price: float) -> float:
        slip = self.config.exit_slippage * self.slippage_mult
        return market_price * (1 - slip) if side == "BUY" else market_price * (1 + slip)
