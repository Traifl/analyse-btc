"""Tests for bot strategy module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.strategy import Signal, XGBoostStrategy, PatternStrategy, ClusterStrategy, CombinedStrategy


@pytest.fixture
def featured_df():
    """DataFrame with all features needed for strategies."""
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
        "ma_7": pd.Series(close).rolling(7).mean().values,
        "ma_25": pd.Series(close).rolling(25).mean().values,
        "ma_99": pd.Series(close).rolling(99).mean().values,
        "rsi": np.random.rand(n) * 100,
        "bb_upper": close + 2 * atr,
        "bb_middle": close,
        "bb_lower": close - 2 * atr,
        "bb_width": 4 * atr,
        "macd": np.random.randn(n) * 100,
        "macd_signal": np.random.randn(n) * 80,
        "macd_hist": np.random.randn(n) * 20,
        "atr": atr,
        "returns": np.random.randn(n) * 0.01,
        "volatility": np.abs(np.random.randn(n) * 0.005) + 0.001,
        "volume_norm": np.random.rand(n) * 3 + 0.5,
    }, index=dates)
    return df.dropna()


def test_signal_dataclass():
    s = Signal(1, 0.8, "test")
    assert s.direction == 1
    assert s.confidence == 0.8
    assert s.name == "test"


def test_xgboost_strategy_buy(featured_df):
    model = MagicMock()
    # Predict price 5% above current
    current = featured_df["close"].iloc[-1]
    model.predict.return_value = np.array([current * 1.05])
    strategy = XGBoostStrategy(model)
    signal = strategy.evaluate(featured_df)
    assert signal.direction == 1
    assert signal.confidence > 0


def test_xgboost_strategy_sell(featured_df):
    model = MagicMock()
    current = featured_df["close"].iloc[-1]
    model.predict.return_value = np.array([current * 0.95])
    strategy = XGBoostStrategy(model)
    signal = strategy.evaluate(featured_df)
    assert signal.direction == -1


def test_xgboost_strategy_hold(featured_df):
    model = MagicMock()
    current = featured_df["close"].iloc[-1]
    model.predict.return_value = np.array([current * 1.001])  # tiny change
    strategy = XGBoostStrategy(model)
    signal = strategy.evaluate(featured_df)
    assert signal.direction == 0


def test_pattern_strategy(featured_df):
    strategy = PatternStrategy()
    signal = strategy.evaluate(featured_df)
    assert signal.name == "pattern"
    assert signal.direction in (-1, 0, 1)


def test_cluster_strategy(featured_df):
    strategy = ClusterStrategy()
    signal = strategy.evaluate(featured_df)
    assert signal.name == "cluster"
    assert signal.direction in (-1, 0, 1)


def test_combined_strategy(featured_df):
    model = MagicMock()
    current = featured_df["close"].iloc[-1]
    model.predict.return_value = np.array([current * 1.05])

    strategies = [
        XGBoostStrategy(model),
        PatternStrategy(),
        ClusterStrategy(),
    ]
    combined = CombinedStrategy(strategies)
    signal, individual = combined.evaluate(featured_df)

    assert signal.name == "combined"
    assert "xgboost" in individual
    assert "pattern" in individual
    assert "cluster" in individual


def test_combined_strategy_custom_weights(featured_df):
    model = MagicMock()
    current = featured_df["close"].iloc[-1]
    model.predict.return_value = np.array([current * 1.05])

    strategies = [
        XGBoostStrategy(model),
        PatternStrategy(),
        ClusterStrategy(),
    ]
    combined = CombinedStrategy(
        strategies,
        weights={"xgboost": 1.0, "pattern": 0.0, "cluster": 0.0},
        threshold=0.2,
    )
    signal, _ = combined.evaluate(featured_df)
    # XGBoost predicts up with full weight -> should buy
    assert signal.direction == 1
