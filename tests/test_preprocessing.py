"""Tests for preprocessing module."""

import pytest
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import clean_data, add_technical_features


@pytest.fixture
def raw_ohlcv():
    """Generate synthetic OHLCV data (200 rows)."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 40000 + np.cumsum(np.random.randn(n) * 100)
    return pd.DataFrame({
        "date": dates,
        "open": close - np.random.rand(n) * 50,
        "high": close + np.random.rand(n) * 100,
        "low": close - np.random.rand(n) * 100,
        "close": close,
        "volume": np.random.rand(n) * 1000 + 100,
    })


def test_clean_data_returns_datetime_index(raw_ohlcv):
    df = clean_data(raw_ohlcv)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert "date" not in df.columns


def test_clean_data_no_nans(raw_ohlcv):
    # Inject some NaNs
    raw_ohlcv.loc[5, "close"] = np.nan
    raw_ohlcv.loc[10, "volume"] = np.nan
    df = clean_data(raw_ohlcv)
    assert df[["open", "high", "low", "close", "volume"]].isna().sum().sum() == 0


def test_clean_data_sorted(raw_ohlcv):
    # Shuffle the data
    shuffled = raw_ohlcv.sample(frac=1, random_state=0)
    df = clean_data(shuffled)
    assert df.index.is_monotonic_increasing


def test_add_technical_features_columns(raw_ohlcv):
    df = clean_data(raw_ohlcv)
    df = add_technical_features(df)
    expected_cols = [
        "ma_7", "ma_25", "ma_99", "rsi",
        "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "macd", "macd_signal", "macd_hist",
        "atr", "returns", "log_returns", "volatility", "volume_norm",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"


def test_add_technical_features_values_range(raw_ohlcv):
    df = clean_data(raw_ohlcv)
    df = add_technical_features(df)
    # RSI should be between 0 and 100
    rsi_valid = df["rsi"].dropna()
    assert rsi_valid.min() >= 0
    assert rsi_valid.max() <= 100
    # Bollinger upper >= middle >= lower
    valid = df.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
    assert (valid["bb_upper"] >= valid["bb_middle"]).all()
    assert (valid["bb_middle"] >= valid["bb_lower"]).all()
