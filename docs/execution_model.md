# Execution Model

Realized PnL uses **actual entry and exit fill prices** only.

## Long (BUY)

```
gross_pnl = (exit_price - entry_price) * quantity
net_pnl   = gross_pnl - entry_fee - exit_fee - slippage_cost
```

## Short (SELL)

```
gross_pnl = (entry_price - exit_price) * quantity
net_pnl   = gross_pnl - entry_fee - exit_fee - slippage_cost
```

## Invariants

- `equity = cash + position * mark_price`
- Unrealized PnL uses current price only
- Realized PnL computed only after close
- Future-return / target columns never influence PnL

## Cost Model

Configurable via `CostConfig`: maker/taker fees, entry/exit slippage, spread.
Stress mode multiplies fees ×1.5 and slippage ×2.
