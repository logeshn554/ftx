# Reward Function

Default: `portfolio_reward` in `src/trade/env/rewards.py`

```
reward = net_return
         - drawdown_penalty * max(0, Δdrawdown)
         - turnover_penalty * turnover / equity
         - risk_penalty
```

- `net_return`: incremental equity change after fees and slippage
- Clipped to [-1, 1] for stable RL training
- HOLD receives zero turnover penalty when no transaction occurs

Coefficients: `training.reward_drawdown_penalty`, `training.reward_turnover_penalty`, `training.reward_risk_penalty`
