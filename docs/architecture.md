# Self-Evolving Trading Platform Architecture

See also: [architecture_baseline.md](architecture_baseline.md), [data_contract.md](data_contract.md).

## Pipeline

```
Market Data → Validation → Features → Regime → Strategies → Strategy Selection
→ EV Engine → Cost Filter → Risk / Survival → Position Sizing → Execution
→ Ledger PnL → Immutable Experience → Statistical Analysis → Evolution
```

## Core Principle

The live **champion is immutable**. Trade outcomes flow to the experience store;
candidates are trained, walk-forward validated, stress-tested, and promoted only
through `ChampionSelector`.

## Key Modules

| Module | Path | Role |
|--------|------|------|
| Ledger | `src/trade/execution/ledger.py` | Canonical PnL |
| Decision Engine | `src/trade/intelligence/decision_engine.py` | Single decision path |
| Target Engine | `src/trade/intelligence/target_engine.py` | Cost-aware TP/SL |
| Evolution Orchestrator | `src/trade/evolution/orchestrator.py` | Self-evolution loop |
| Drift Detector | `src/trade/learning/drift.py` | Distribution monitoring |

## Paper Trading

`scripts/server.py` uses `PaperTradingSession` which delegates to the canonical
ledger and decision engine. Dashboard metrics originate from ledger snapshots.
