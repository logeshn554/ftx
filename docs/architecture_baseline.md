# Architecture Baseline Audit

Branch: `self-evolving-v2`  
Date: 2026-09-01  
Scope: Read-only audit before remediation. No behavior changes in this document.

## Executive Summary

The repository is a hybrid of a research-grade `src/trade/` package and a legacy
dashboard runtime in `scripts/server.py` + `scripts/webrl_engine.py`. The package
layer already contains causal accounting, data contracts, walk-forward validation,
and candidate generation. The live paper-trading path still bypasses several of
those guarantees.

**Critical gaps before remediation:**

| Severity | Issue | Location |
|----------|-------|----------|
| CRITICAL | Gross TP triggers close while net PnL may be negative | `scripts/server.py` |
| CRITICAL | TP/SL thresholds (0.015%) below round-trip cost (~0.20%) | `scripts/server.py` |
| CRITICAL | Account-death loop mutates live policy weights | `scripts/server.py`, `webrl_engine.py` |
| HIGH | Duplicate decision paths (server vs `DecisionPipeline`) | `scripts/server.py` |
| HIGH | `run_100_attempt_macro_analysis` recalibrates live policy | `webrl_engine.py` |
| HIGH | `check_equity_boundary` mutates `base_policy` on profit/ruin | `webrl_engine.py` |
| MEDIUM | Dashboard shows `estimated_loss_reduction` as empirical | `dashboard/app.js` |
| MEDIUM | Q-table still updates on every trade (research signal, not champion) | `webrl_engine.py` |

## Data Flow

```
Market (Binance / Yahoo / seed)
    → scripts/server.py price_history + indicators
    → trade.data.features.FeatureEngine (backtest/train only)
    → trade.data.contract (OBSERVATION_FEATURES firewall)
    → trade.env.TradingEnv (RL training/backtest)
```

**Leakage controls present:** `trade.data.contract.observation_columns`,
`FeatureEngine.extract_observation`, tests in `tests/test_data/test_leakage.py`.

**Leakage risk:** Server does not use `FeatureEngine`; indicators are computed
inline from price history (causal). No `return_1_1m` in live path.

## Training Flow

```
scripts/train.py
    → trade.agent.trainer
    → trade.env.TradingEnv (PositionAccounting, portfolio_reward)
    → trade.experience.collector / store (episodes)
```

Scalers/features: `FeatureEngine` uses rolling past-only indicators. Walk-forward
splits are chronological (`validation/walk_forward.py`).

## Backtest / Validation Flow

```
scripts/backtest.py / scripts/evaluate.py
    → trade.validation.backtester
    → trade.validation.walk_forward.WalkForwardValidator
    → trade.validation.comparator.ModelComparator
    → trade.validation.gatekeeper.Gatekeeper
```

Walk-forward uses train/validation/test slices; OOS evaluation on test window only.

## Inference / Execution Flow (Package)

```
trade.agent.inference
    → trade.intelligence.decision.DecisionPipeline
    → trade.intelligence.expected_value.ExpectedValueFilter
    → trade.risk.engine.RiskEngine
    → trade.execution.paper / live / shadow
```

## Inference / Execution Flow (Dashboard Server — Legacy)

```
scripts/server.py trading_loop
    → inline RSI/EMA signal detection
    → scripts/webrl_engine.WebRLEngine.evaluate_trade
    → inline position dict + manual PnL
    → webrl.on_loss / on_win (Q-table update)
```

**This path does NOT use:** `DecisionPipeline`, `TargetEngine`, `Ledger`,
`SurvivalController`, or `position_sizing` from package layer.

## Reward Flow

| Context | Reward | File |
|---------|--------|------|
| RL env | `portfolio_reward` (net return − drawdown − turnover) | `env/rewards.py` |
| Q-learning | `pnl_pct / 1.5` clipped | `webrl_engine.py` |
| WebRL ORM | deferred (comment only) | `webrl_engine.py` |

## Model Promotion Flow

```
CandidateGenerator → (train) → WalkForwardValidator → ModelComparator → Gatekeeper
```

Champion registry: `trade.model_management.registry`, rollback in `rollback.py`.

**Not wired to dashboard server.** Server has autonomous "Generation #" retrain
on balance < $2 which is NOT promotion-gated.

## Trade Lifecycle Map

### Trade Initiation

| Path | File | Function |
|------|------|----------|
| Dashboard | `scripts/server.py` | `trading_loop` ~L576 |
| Package paper | `trade/execution/paper.py` | `PaperBroker` |
| Env | `trade/env/trading_env.py` | `_execute_action` |
| WebRL eval | `webrl_engine.py` | `evaluate_trade` |

### Trade Close

| Path | File | Notes |
|------|------|-------|
| Dashboard | `scripts/server.py` | TP/SL/time on **gross** pct |
| Accounting | `trade/execution/accounting.py` | Price-based, correct |
| Env | `trade/env/trading_env.py` | Via `PositionAccounting.close` |

### PnL Calculation

| Path | Uses future columns? | Correct BUY/SELL? |
|------|---------------------|-------------------|
| `accounting.py` | No | Yes |
| `server.py` | No (uses live price) | Yes, but gross TP logic wrong |
| `webrl_engine.py` | No | Receives pre-computed ctx |

### Fees / Slippage

| Path | Entry fee | Exit fee | Slippage |
|------|-----------|----------|----------|
| `accounting.py` | Explicit | Explicit | Adverse fill adjustment |
| `server.py` | 0.1% of allocated | 0.1% of allocated | None |
| `cost_model.py` | **Missing** | **Missing** | **Missing** |

### Model Parameter Mutation

| Trigger | Mutates live policy? | File |
|---------|---------------------|------|
| `on_loss` / `on_win` | Q-table only; policy adapter NOT called | `webrl_engine.py` |
| `adapt_on_loss` / `adapt_on_win` | Yes (if called) | `webrl_engine.py` — dead code |
| `run_100_attempt_macro_analysis` | Yes (`base_policy`) | `webrl_engine.py` |
| `check_equity_boundary` | Yes | `webrl_engine.py` |
| Account death in server | Triggers macro analysis | `server.py` |
| `CandidateGenerator` | No (immutable copy) | `evolution/candidate_generator.py` |

## Duplicate Decision Paths

1. **`DecisionPipeline`** (`intelligence/decision.py`) — EV + confidence gates
2. **`WebRLEngine.evaluate_trade`** — Q + MuZero + GRPO composite
3. **`server.py` inline signals** — RSI/EMA rules independent of strategies package
4. **`trade/strategies/*`** — Not used by server

Canonical target: single `DecisionEngine` consumed by server and scripts.

## Components Present vs Required

| Required | Status |
|----------|--------|
| `execution/position.py` | Missing |
| `execution/cost_model.py` | Missing |
| `execution/ledger.py` | Missing (`accounting.py` exists) |
| `data/schema.py` | Partial (`contract.py` exists) |
| `intelligence/target_engine.py` | Missing |
| `intelligence/decision_engine.py` | Missing (`decision.py` partial) |
| `evolution/experience_store.py` (trades) | Missing (`experience/store.py` is RL episodes) |
| `evolution/evaluator.py` | Missing |
| `evolution/champion_selector.py` | Missing |
| `evolution/orchestrator.py` | Missing |
| `learning/drift.py` | Missing |
| `evolution/llm_researcher.py` | Missing |

## Test Coverage (Baseline)

57 tests passing. Categories covered: accounting, leakage, env, walk-forward
isolation, candidate generation, comparator, risk engine, intelligence gates.

Missing test categories: ledger reconciliation, target engine, decision engine
integration, champion immutability under server events, drift detection.

## Remediation Priority

1. Canonical execution layer (`position`, `cost_model`, `ledger`)
2. Cost-aware target engine + decision engine
3. Disable live policy mutation in WebRL
4. Integrate server with canonical pipeline
5. Evolution orchestrator + champion selector
6. Dashboard telemetry from ledger only
