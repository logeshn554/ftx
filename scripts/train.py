"""CLI: Train a new PPO model."""

import argparse
import datetime as dt
import sys

from trade.core.config import build_config
from trade.core.logging import setup_logging, get_logger
from trade.core.types import ModelVersion, ModelStage
from trade.data.sources.yahoo import YahooDataSource
from trade.data.validation import DataValidator
from trade.data.features import FeatureEngine
from trade.env.trading_env import TradingEnv
from trade.agent.trainer import AgentTrainer
from trade.model_management.registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description="Train a new PPO trading model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to train on")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps")
    parser.add_argument("--version", type=str, default="0.1.0", help="Model version (major.minor.patch)")
    args = parser.parse_args()

    config = build_config(args.config)
    if args.timesteps:
        config.training.total_timesteps = args.timesteps
    setup_logging(config.log_level, config.log_format)
    log = get_logger("train")

    symbol = args.symbol or config.data.symbols[0]

    # Parse version
    parts = args.version.split(".")
    version = ModelVersion(
        major=int(parts[0]),
        minor=int(parts[1]) if len(parts) > 1 else 0,
        patch=int(parts[2]) if len(parts) > 2 else 0,
        stage=ModelStage.CANDIDATE,
    )

    log.info("Training model", symbol=symbol, version=version.tag, timesteps=config.training.total_timesteps)

    # Prepare data
    source = YahooDataSource(cache_dir=config.data.cache_dir)
    validator = DataValidator()
    feature_engine = FeatureEngine(feature_window=config.data.feature_window)

    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=config.data.lookback_days)

    df = source.fetch_ohlcv(symbol, start_date, end_date, config.data.timeframe)
    df = validator.clean(df)
    features_df = feature_engine.compute_features(df)
    feature_cols = feature_engine.get_feature_columns()

    # Split train/eval (80/20)
    split_idx = int(len(features_df) * 0.8)
    train_df = features_df.iloc[:split_idx].copy()
    eval_df = features_df.iloc[split_idx:].copy()

    log.info("Data split", train_bars=len(train_df), eval_bars=len(eval_df))

    # Create environments
    train_env = TradingEnv(
        features_df=train_df,
        feature_columns=feature_cols,
        initial_capital=config.trading.initial_capital,
        commission_pct=config.trading.commission_pct,
        slippage_pct=config.trading.slippage_pct,
        feature_window=config.data.feature_window,
        reward_function=config.training.reward_function,
        reward_drawdown_penalty=config.training.reward_drawdown_penalty,
        reward_turnover_penalty=config.training.reward_turnover_penalty,
        reward_risk_penalty=config.training.reward_risk_penalty,
    )

    eval_env = TradingEnv(
        features_df=eval_df,
        feature_columns=feature_cols,
        initial_capital=config.trading.initial_capital,
        commission_pct=config.trading.commission_pct,
        slippage_pct=config.trading.slippage_pct,
        feature_window=config.data.feature_window,
        reward_function=config.training.reward_function,
        reward_drawdown_penalty=config.training.reward_drawdown_penalty,
        reward_turnover_penalty=config.training.reward_turnover_penalty,
        reward_risk_penalty=config.training.reward_risk_penalty,
    )

    # Train
    trainer = AgentTrainer(config)
    result = trainer.train(
        train_env=train_env,
        eval_env=eval_env,
        model_version=version,
    )

    # Register in model registry
    registry = ModelRegistry(registry_dir=config.model.registry_dir)
    registry.register(
        version=version,
        checkpoint_path=result.checkpoint_path,
        metrics=result.final_metrics,
    )

    log.info(
        "Training complete",
        version=version.tag,
        time=f"{result.training_time_seconds:.0f}s",
        checkpoint=result.checkpoint_path,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
