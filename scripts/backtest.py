"""CLI: Backtest a model against historical data."""

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


def main():
    parser = argparse.ArgumentParser(description="Backtest a model against historical data")
    parser.add_argument("model_path", type=str, help="Path to saved model checkpoint")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--days", type=int, default=None, help="Backtest period in days")
    args = parser.parse_args()

    config = build_config(args.config)
    setup_logging(config.log_level, config.log_format)
    log = get_logger("backtest")

    symbol = args.symbol or config.data.symbols[0]
    days = args.days or config.data.lookback_days

    log.info("Running backtest", model=args.model_path, symbol=symbol, days=days)

    # Prepare data
    source = YahooDataSource(cache_dir=config.data.cache_dir)
    validator = DataValidator()
    feature_engine = FeatureEngine(feature_window=config.data.feature_window)

    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=days)

    df = source.fetch_ohlcv(symbol, start_date, end_date, config.data.timeframe)
    df = validator.clean(df)
    features_df = feature_engine.compute_features(df)
    feature_cols = feature_engine.get_feature_columns()

    # Run backtest
    backtester = Backtester(
        initial_capital=config.trading.initial_capital,
        commission_pct=config.trading.commission_pct,
        slippage_pct=config.trading.slippage_pct,
        feature_window=config.data.feature_window,
    )

    result = backtester.run(
        model_path=args.model_path,
        features_df=features_df,
        feature_columns=feature_cols,
    )

    # Print results
    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS — {symbol}")
    print("=" * 60)
    print(f"  Period:           {result.start_date} → {result.end_date}")
    print(f"  Initial Capital:  ${result.initial_capital:,.2f}")
    print(f"  Final Capital:    ${result.final_capital:,.2f}")
    print(f"  Total Return:     {result.total_return:.2%}")
    print(f"  Sharpe Ratio:     {result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown:     {result.max_drawdown:.2%}")
    print(f"  Win Rate:         {result.win_rate:.1%}")
    print(f"  Profit Factor:    {result.profit_factor:.2f}")
    print(f"  Total Trades:     {result.total_trades}")
    print(f"  Transaction Costs: ${result.transaction_costs:,.2f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
