"""CLI: Bootstrap a baseline model if none exists.

This script trains a minimal untrained PPO model to produce a valid
checkpoint file structure. Used to get the system to a runnable state.
"""

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Bootstrap a baseline model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to train on")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Timesteps for bootstrap (default 50k)")
    args = parser.parse_args()

    config = build_config(args.config)
    setup_logging(config.log_level, config.log_format)
    log = get_logger("bootstrap")

    symbol = args.symbol or config.data.symbols[0]

    # Check if checkpoint already exists
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    baseline_path = checkpoint_dir / "model_v0.1.0.zip"

    if baseline_path.exists():
        log.info("✓ Baseline model already exists at %s", baseline_path)
        return 0

    log.info("🚀 Bootstrapping baseline model for %s (%d timesteps)", symbol, args.timesteps)

    try:
        # Download minimal data (90 days)
        log.info("Step 1/4: Downloading data...")
        source = YahooDataSource(cache_dir=config.data.cache_dir)
        end_date = dt.date.today()
        start_date = end_date - dt.timedelta(days=90)

        df = source.fetch_ohlcv(symbol, start=start_date, end=end_date, timeframe="1d")
        if df is None or len(df) < 30:
            log.error("Failed to download sufficient data for %s", symbol)
            return 1

        # Validate and engineer features
        log.info("Step 2/4: Validating and computing features...")
        validator = DataValidator()
        val_result = validator.validate(df, symbol)
        if not val_result.passed:
            log.error("Data validation failed: %s", val_result.issues)
            return 1

        df_clean = validator.clean(df)
        feature_engine = FeatureEngine(feature_window=config.data.feature_window)
        df_features = feature_engine.compute_features(df_clean)

        # Create env and train
        log.info("Step 3/4: Creating environment...")
        env = TradingEnv(
            ohlcv_df=df_features,
            feature_columns=[c for c in df_features.columns if c in config.data.feature_window],
            initial_cash=config.trading.initial_capital,
            commission_pct=config.trading.commission_pct,
        )

        log.info("Step 4/4: Training baseline model (%d timesteps)...", args.timesteps)
        version = ModelVersion(major=0, minor=1, patch=0, stage=ModelStage.CANDIDATE)
        trainer = AgentTrainer(config=config.training)
        model = trainer.train(env, total_timesteps=args.timesteps, model_name=version.tag)

        # Save checkpoint
        model.save(str(baseline_path.with_suffix("")))
        log.info("✓ Baseline model saved to %s", baseline_path)

        # Register in model registry
        registry = ModelRegistry()
        registry.register(
            version=version,
            model_path=str(baseline_path),
            stage=ModelStage.PAPER_TRADING,
            metadata={"bootstrap": True, "symbol": symbol, "timesteps": args.timesteps},
        )
        log.info("✓ Model registered in MLflow registry as %s", version.tag)

        return 0

    except Exception as e:
        log.error("Bootstrap failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
