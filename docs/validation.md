# Validation

## Walk-Forward

Chronological windows: TRAIN → VALIDATE (optional) → TEST (OOS).

Per window: net return, Sharpe, Sortino, max drawdown, profit factor, expectancy, fees, slippage.

Aggregates: mean, median, std, worst window, positive-window ratio.

## Promotion Gates

- OOS Sharpe improvement margin
- Profit factor minimum
- Expectancy gain
- Positive walk-forward ratio
- Max drawdown limit
- Cost stress net return > 0
- Minimum trade count

See `ChampionSelector` and `ModelComparator`.
