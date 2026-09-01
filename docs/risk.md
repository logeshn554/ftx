# Risk

## Survival Controller

States: NORMAL → CAUTION → DEFENSIVE → HALTED

HALTED blocks all new trades; RL cannot override.

## Position Sizing

`position_size()` scales by edge, confidence, volatility, drawdown, stop distance.
Hard caps: `max_risk_per_trade`, `max_position_pct`.

## Risk Engine

`trade.risk.engine.RiskEngine` — final gate before order execution.
