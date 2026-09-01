"""Centralized configuration with Pydantic settings and YAML overlay loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Config sub-sections
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    """Data pipeline settings."""

    symbols: list[str] = Field(default=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
    timeframe: str = "1d"
    lookback_days: int = 730  # 2 years
    feature_window: int = 30  # observation window in bars
    cache_dir: str = "data_cache"

    # Validation thresholds
    max_price_jump_pct: float = 20.0
    min_volume: int = 100
    max_missing_bars_pct: float = 5.0


class TradingConfig(BaseModel):
    """Trading / execution settings."""

    initial_capital: float = 100_000.0
    commission_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage
    default_position_size_pct: float = 10.0  # % of capital per trade
    mode: str = "paper"  # paper | shadow | live


class RiskConfig(BaseModel):
    """Risk management limits."""

    max_position_pct: float = 20.0  # max % of capital in single position
    max_daily_loss_pct: float = 5.0  # daily stop-loss threshold
    max_leverage: float = 1.0  # no leverage by default
    max_order_value: float = 50_000.0  # absolute cap on single order
    max_open_positions: int = 10
    min_data_freshness_seconds: float = 300.0  # 5 min staleness limit
    circuit_breaker_cooldown_seconds: float = 3600.0  # 1 hour


class TrainingConfig(BaseModel):
    """RL training hyperparameters."""

    algorithm: str = "PPO"
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_function: str = "risk_adjusted"  # sharpe | pnl | risk_adjusted | differential_sharpe
    policy_architecture: str = "mlp"  # mlp | cnn | lstm

    # Feature extractor
    net_arch_pi: list[int] = Field(default=[256, 128, 64])
    net_arch_vf: list[int] = Field(default=[256, 128, 64])

    # Checkpointing
    eval_freq: int = 10_000
    checkpoint_freq: int = 50_000


class ModelConfig(BaseModel):
    """Model management settings."""

    registry_dir: str = "mlflow_registry"
    min_sharpe_improvement: float = 0.1  # V2 must beat V1 by at least this
    max_drawdown_regression: float = 0.02  # V2 drawdown can't be worse by more than 2%
    min_win_rate: float = 0.45
    paper_trading_days: int = 30
    shadow_trading_days: int = 14


class LearningConfig(BaseModel):
    """Learning agent / retraining settings."""

    performance_window_days: int = 30
    sharpe_threshold: float = 0.5  # retrain if rolling Sharpe drops below this
    retrain_cooldown_hours: int = 24
    max_experience_age_days: int = 180
    scheduled_retrain_days: int = 30  # retrain every N days regardless


class APIConfig(BaseModel):
    """FastAPI server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:8000"])


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig(BaseSettings):
    """Root application configuration. Loads from env vars and YAML overlay."""

    model_config = {"env_prefix": "TRADE_", "env_nested_delimiter": "__"}

    # Sub-sections
    data: DataConfig = Field(default_factory=DataConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    # Global
    db_url: str = "sqlite:///trade.db"
    log_level: str = "INFO"
    log_format: str = "dev"  # dev | json
    base_dir: str = "."


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as dict."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def build_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Build AppConfig by merging defaults → YAML → env vars → overrides.

    Priority (lowest to highest):
        1. Pydantic defaults
        2. YAML file values
        3. Environment variables (TRADE__ prefixed)
        4. Explicit overrides dict
    """
    yaml_data: dict[str, Any] = {}

    # Try explicit path, then TRADE_CONFIG_PATH env var, then default locations
    if config_path:
        yaml_data = load_yaml_config(config_path)
    elif env_path := os.environ.get("TRADE_CONFIG_PATH"):
        yaml_data = load_yaml_config(env_path)
    else:
        for candidate in ["config/default.yaml", "config.yaml"]:
            loaded = load_yaml_config(candidate)
            if loaded:
                yaml_data = loaded
                break

    if overrides:
        _deep_merge(yaml_data, overrides)

    return AppConfig(**yaml_data)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
