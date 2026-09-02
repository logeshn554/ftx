"""Shared test fixtures."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import pathlib
import shutil
import tempfile
from trade.core.config import AppConfig, build_config
from trade.core.types import Order, OrderSide, PortfolioState, Position


@pytest.fixture
def tmp_path():
    import uuid
    base = pathlib.Path(__file__).parent.parent / ".tmp_tests"
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"run_{uuid.uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=True)
    yield target
    try:
        shutil.rmtree(target, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def config() -> AppConfig:
    """Test configuration with reduced settings for speed."""
    return build_config(overrides={
        "db_url": "sqlite:///:memory:",
        "training": {
            "total_timesteps": 1000,
            "eval_freq": 500,
            "checkpoint_freq": 500,
        },
        "data": {
            "lookback_days": 365,
            "feature_window": 10,
        },
    })


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 200
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="B")

    # Random walk price
    returns = np.random.normal(0.0005, 0.02, n_bars)
    close = 100 * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n_bars)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n_bars)))
    open_ = close * (1 + np.random.normal(0, 0.005, n_bars))
    volume = np.random.randint(1_000_000, 10_000_000, n_bars).astype(float)

    df = pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=dates)

    return df


@pytest.fixture
def sample_portfolio() -> PortfolioState:
    """Sample portfolio state for risk engine testing."""
    return PortfolioState(
        cash=80_000.0,
        positions={
            "AAPL": Position(
                symbol="AAPL",
                quantity=100,
                avg_entry_price=150.0,
                current_price=155.0,
                unrealized_pnl=500.0,
            )
        },
        total_equity=95_500.0,
    )


@pytest.fixture
def sample_buy_order() -> Order:
    """Sample buy order for testing."""
    return Order(
        symbol="MSFT",
        side=OrderSide.LONG,
        quantity=50,
    )
