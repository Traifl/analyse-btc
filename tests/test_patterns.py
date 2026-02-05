"""Tests for patterns module."""

import pytest
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.patterns import (
    detect_range, detect_breakout, detect_volume_spike,
    detect_patterns, generate_pattern_report,
)


@pytest.fixture
def featured_df():
    """DataFrame with all required columns for pattern detection."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 40000 + np.cumsum(np.random.randn(n) * 100)
    atr = np.abs(np.random.randn(n) * 200) + 50

    df = pd.DataFrame({
        "open": close - 50,
        "high": close + 100,
        "low": close - 100,
        "close": close,
        "volume": np.random.rand(n) * 1000 + 100,
        "atr": atr,
        "bb_upper": close + 2 * atr,
        "bb_lower": close - 2 * atr,
        "volume_norm": np.random.rand(n) * 3 + 0.5,
        "rsi": np.random.rand(n) * 100,
    }, index=dates)
    return df


def test_detect_range_returns_bool(featured_df):
    result = detect_range(featured_df)
    assert result.dtype == bool
    assert result.sum() > 0  # some periods should be in range


def test_detect_breakout_values(featured_df):
    # Force a breakout_up
    df = featured_df.copy()
    df.iloc[0, df.columns.get_loc("close")] = df.iloc[0]["bb_upper"] + 1000
    result = detect_breakout(df)
    assert result.iloc[0] == "breakout_up"


def test_detect_volume_spike(featured_df):
    result = detect_volume_spike(featured_df)
    assert result.dtype == bool


def test_detect_patterns_adds_columns(featured_df):
    result = detect_patterns(featured_df)
    for col in ["is_range", "breakout", "fib_retracement", "volume_spike"]:
        assert col in result.columns


def test_detect_patterns_missing_columns():
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError, match="Missing required columns"):
        detect_patterns(df)


def test_generate_pattern_report(featured_df):
    df = detect_patterns(featured_df)
    report = generate_pattern_report(df)
    assert "Pattern Analysis Report" in report
    assert "Range" in report
    assert "Breakout" in report
