"""Tests for data validation."""

import numpy as np
import pandas as pd
import pytest

from trade.data.validation import DataValidator


class TestDataValidator:
    """Test data validation checks."""

    def test_valid_data_passes(self, sample_ohlcv):
        """Clean data passes validation."""
        validator = DataValidator()
        result = validator.validate(sample_ohlcv, "TEST")
        assert result.passed
        assert result.error_count == 0

    def test_empty_dataframe_fails(self):
        """Empty DataFrame fails validation."""
        validator = DataValidator()
        result = validator.validate(pd.DataFrame(), "TEST")
        assert not result.passed

    def test_missing_columns_fails(self):
        """Missing required columns fails validation."""
        validator = DataValidator()
        df = pd.DataFrame({"open": [1, 2], "close": [1, 2]})
        result = validator.validate(df, "TEST")
        assert not result.passed

    def test_nan_values_detected(self, sample_ohlcv):
        """NaN values are detected."""
        validator = DataValidator()
        df = sample_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("close")] = np.nan

        result = validator.validate(df, "TEST")
        assert any("NaN" in str(i.message) for i in result.issues)

    def test_negative_prices_detected(self, sample_ohlcv):
        """Negative prices are detected as errors."""
        validator = DataValidator()
        df = sample_ohlcv.copy()
        df.iloc[10, df.columns.get_loc("close")] = -1.0

        result = validator.validate(df, "TEST")
        assert not result.passed
        assert any("non-positive" in str(i.message) for i in result.issues)

    def test_clean_removes_nans(self, sample_ohlcv):
        """Clean method removes NaN rows."""
        validator = DataValidator()
        df = sample_ohlcv.copy()
        df.iloc[5, df.columns.get_loc("close")] = np.nan
        df.iloc[10, df.columns.get_loc("open")] = np.nan

        cleaned = validator.clean(df)
        assert not cleaned["close"].isna().any()
        assert not cleaned["open"].isna().any()
        assert len(cleaned) == len(df) - 2

    def test_clean_sorts_index(self, sample_ohlcv):
        """Clean method sorts the index."""
        validator = DataValidator()
        df = sample_ohlcv.iloc[::-1]  # Reverse order

        cleaned = validator.clean(df)
        assert cleaned.index.is_monotonic_increasing
