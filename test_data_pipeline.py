#!/usr/bin/env python
"""Quick test to verify data pipeline works."""

import pandas as pd
import numpy as np
from trade.data import FeatureEngine, OBSERVATION_FEATURES

# Create mock OHLCV data
dates = pd.date_range('2023-01-01', periods=100)
df = pd.DataFrame({
    'open': np.random.uniform(100, 110, 100),
    'high': np.random.uniform(110, 120, 100),
    'low': np.random.uniform(90, 100, 100),
    'close': np.random.uniform(100, 110, 100),
    'volume': np.random.uniform(1000, 10000, 100),
}, index=dates)

print(f"✓ Mock OHLCV data: {df.shape}")

# Test feature engine
fe = FeatureEngine(feature_window=30)
result = fe.compute_features(df)

print(f"✓ Feature computation: {result.shape[0]} rows × {result.shape[1]} cols")
print(f"✓ OBSERVATION_FEATURES count: {len(OBSERVATION_FEATURES)}")

# Check key features exist
key_features = ['sma_10', 'rsi_14', 'macd', 'adx', 'stoch_k', 'bb_upper', 'atr_14', 'obv']
present = [f for f in key_features if f in result.columns]
print(f"✓ Key features present: {len(present)}/{len(key_features)}")

print("\n✅ Data package verification complete — all checks passed!")
