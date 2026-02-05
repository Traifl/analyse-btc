"""Tests for backtesting module."""

import pytest
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtesting import generate_signals, backtest_strategy, generate_backtest_report


@pytest.fixture
def sample_data():
    """Create sample data for backtesting tests."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 40000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame({"close": close}, index=dates)
    return df


def test_generate_signals_shape(sample_data):
    predictions = sample_data["close"].values + np.random.randn(len(sample_data)) * 500
    signals = generate_signals(sample_data, predictions, sample_data.index)
    assert "signal" in signals.columns
    assert "close" in signals.columns
    assert "predicted" in signals.columns
    assert len(signals) == len(sample_data)


def test_generate_signals_values(sample_data):
    # Predictions always above current price -> all buy signals
    predictions = sample_data["close"].values + 1000
    signals = generate_signals(sample_data, predictions, sample_data.index)
    assert (signals["signal"] == 1).all()


def test_backtest_strategy_basic(sample_data):
    predictions = sample_data["close"].values + np.random.randn(len(sample_data)) * 500
    signals = generate_signals(sample_data, predictions, sample_data.index)
    results = backtest_strategy(signals, initial_capital=10000)

    assert "final_value" in results
    assert "profit" in results
    assert "total_return_pct" in results
    assert "max_drawdown_pct" in results
    assert "sharpe_ratio" in results
    assert results["initial_capital"] == 10000


def test_backtest_no_trades():
    """All hold signals -> no trades, capital unchanged."""
    dates = pd.date_range("2024-01-01", periods=10, freq="h")
    signals = pd.DataFrame({
        "close": [40000] * 10,
        "signal": [0] * 10,
    }, index=dates)
    results = backtest_strategy(signals, initial_capital=10000)
    assert results["n_trades"] == 0
    assert results["final_value"] == 10000


def test_backtest_single_trade():
    """Buy then sell, verify trade count."""
    dates = pd.date_range("2024-01-01", periods=5, freq="h")
    signals = pd.DataFrame({
        "close": [40000, 40000, 41000, 41000, 41000],
        "signal": [1, 0, -1, 0, 0],
    }, index=dates)
    results = backtest_strategy(signals, initial_capital=10000, fee_pct=0)
    assert results["n_trades"] == 2  # 1 buy + 1 sell
    # Should have profited from 40000 -> 41000
    assert results["profit"] > 0


def test_generate_backtest_report():
    results = {
        "initial_capital": 10000,
        "final_value": 11000,
        "profit": 1000,
        "total_return_pct": 10.0,
        "max_drawdown_pct": -5.0,
        "sharpe_ratio": 1.5,
        "n_trades": 20,
    }
    report = generate_backtest_report(results)
    assert "Backtest Report" in report
    assert "$10,000.00" in report
    assert "10.00%" in report
