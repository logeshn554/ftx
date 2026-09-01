"""CLI: Compare two model versions (V1 champion vs V2 challenger)."""

import argparse
import datetime as dt
import sys

from trade.core.config import build_config
from trade.core.logging import setup_logging, get_logger
from trade.core.types import ModelVersion
from trade.data.sources.yahoo import YahooDataSource
from trade.data.validation import DataValidator
from trade.data.features import FeatureEngine
from trade.validation.backtester import Backtester
from trade.validation.comparator import ModelComparator


def main():
    parser = argparse.ArgumentParser(description="Compare two model versions")
    parser.add_argument("champion_path", help="Path to champion (V1) model")
    parser.add_argument("challenger_path", help="Path to challenger (V2) model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()

    config = build_config(args.config)
    setup_logging(config.log_level, config.log_format)
    log = get_logger("evaluate")

    symbol = args.symbol or config.data.symbols[0]

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

    backtester = Backtester(
        initial_capital=config.trading.initial_capital,
        commission_pct=config.trading.commission_pct,
        slippage_pct=config.trading.slippage_pct,
        feature_window=config.data.feature_window,
    )

    champion_v = ModelVersion(major=0, minor=1, patch=0)
    challenger_v = ModelVersion(major=0, minor=2, patch=0)

    log.info("Backtesting champion...")
    champ_result = backtester.run(args.champion_path, features_df, feature_cols, champion_v)

    log.info("Backtesting challenger...")
    chal_result = backtester.run(args.challenger_path, features_df, feature_cols, challenger_v)

    # Compare
    comparator = ModelComparator(
        min_sharpe_improvement=config.model.min_sharpe_improvement,
        max_drawdown_regression=config.model.max_drawdown_regression,
        min_win_rate=config.model.min_win_rate,
    )
    result = comparator.compare(champ_result, chal_result)

    # Print comparison
    print("\n" + "=" * 60)
    print(f"  MODEL COMPARISON — {symbol}")
    print("=" * 60)
    print(f"  Champion:    {champion_v.tag}")
    print(f"  Challenger:  {challenger_v.tag}")
    print(f"  VERDICT:     {result.verdict.value}")
    print("-" * 60)
    print(f"  {'Metric':<25} {'Champion':>12} {'Challenger':>12} {'Delta':>10}")
    print("-" * 60)

    for metric in ["sharpe_ratio", "max_drawdown", "win_rate", "total_return", "profit_factor"]:
        c_val = result.champion_metrics.get(metric, 0)
        h_val = result.challenger_metrics.get(metric, 0)
        delta = h_val - c_val
        print(f"  {metric:<25} {c_val:>12.4f} {h_val:>12.4f} {delta:>+10.4f}")

    print("=" * 60)
    print(f"\n{result.notes}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
