"""CLI: Download and prepare historical market data."""

import argparse
import datetime as dt
import sys

from trade.core.config import build_config
from trade.core.logging import setup_logging, get_logger
from trade.data.sources.yahoo import YahooDataSource
from trade.data.validation import DataValidator
from trade.data.features import FeatureEngine
from trade.data.storage import DataStore


def main():
    parser = argparse.ArgumentParser(description="Download and prepare historical data")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--symbols", nargs="+", default=None, help="Override symbols")
    parser.add_argument("--days", type=int, default=None, help="Override lookback days")
    args = parser.parse_args()

    config = build_config(args.config)
    setup_logging(config.log_level, config.log_format)
    log = get_logger("seed_data")

    symbols = args.symbols or config.data.symbols
    lookback = args.days or config.data.lookback_days

    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=lookback)

    log.info("Seeding data", symbols=symbols, start=str(start_date), end=str(end_date))

    # Initialize components
    source = YahooDataSource(cache_dir=config.data.cache_dir)
    validator = DataValidator(
        max_price_jump_pct=config.data.max_price_jump_pct,
        min_volume=config.data.min_volume,
        max_missing_bars_pct=config.data.max_missing_bars_pct,
    )
    feature_engine = FeatureEngine(feature_window=config.data.feature_window)
    store = DataStore(db_url=config.db_url)

    success = 0
    failed = 0

    for symbol in symbols:
        try:
            log.info("Processing", symbol=symbol)

            # Fetch
            df = source.fetch_ohlcv(symbol, start_date, end_date, config.data.timeframe)
            log.info("Fetched", symbol=symbol, bars=len(df))

            # Validate
            result = validator.validate(df, symbol)
            store.log_validation(
                symbol=symbol,
                passed=result.passed,
                error_count=result.error_count,
                warning_count=result.warning_count,
                details=str([str(i) for i in result.issues]),
            )

            if not result.passed:
                log.warning("Validation failed", symbol=symbol, errors=result.error_count)
                df = validator.clean(df)

            # Store OHLCV
            saved = store.save_ohlcv(df, symbol, config.data.timeframe)
            log.info("Saved OHLCV", symbol=symbol, rows=saved)

            # Compute features
            features_df = feature_engine.compute_features(df)
            log.info("Features computed", symbol=symbol, features=len(features_df.columns))

            success += 1

        except Exception as exc:
            log.error("Failed to process", symbol=symbol, error=str(exc))
            failed += 1

    log.info(
        "Data seeding complete",
        success=success,
        failed=failed,
        total=len(symbols),
    )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
