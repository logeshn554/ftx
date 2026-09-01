"""Configurable transaction cost model with stress multipliers."""

from __future__ import annotations

from dataclasses import dataclass


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
