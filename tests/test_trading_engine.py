"""Tests for trading engine module — tests component wiring without live connections."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.trading_engine import TradingEngine
from bot.state import BotState


@pytest.fixture
def state(tmp_path):
    return BotState(initial_capital=10000, state_file=str(tmp_path / "test_state.json"))


@pytest.fixture
def sample_candle():
    return {
        "open": 50000.0,
        "high": 50500.0,
        "low": 49500.0,
        "close": 50200.0,
        "volume": 100.0,
        "close_time": 1704067200000,
    }


@pytest.fixture
def warm_buffer():
    """Pre-built raw OHLCV buffer + featured buffer."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 50000 + np.cumsum(np.random.randn(n) * 100)

    from src.preprocessing import add_technical_features
    raw = pd.DataFrame({
        "open": close - 50,
        "high": close + 100,
        "low": close - 100,
        "close": close,
        "volume": np.random.rand(n) * 1000 + 100,
    }, index=dates)
    featured = add_technical_features(raw.copy()).dropna()
    return raw, featured


def test_engine_init(state):
    engine = TradingEngine(mode="paper", state=state)
    assert engine.mode == "paper"
    assert engine.is_running is False
    assert engine.state is state


def test_process_candle(state, warm_buffer, sample_candle):
    """Test that _process_candle runs without error given proper setup."""
    raw, featured = warm_buffer
    engine = TradingEngine(mode="paper", state=state)
    engine.raw_buffer = raw.copy()
    engine.candle_buffer = featured

    # Mock model
    model = MagicMock()
    model.predict.return_value = np.array([50300.0])
    engine.model = model

    # Setup strategies
    engine._setup_strategies()
    engine._setup_executor()

    # Process candle
    engine._process_candle(sample_candle)

    # Check state was updated
    assert state.get("current_price") == 50200.0
    assert state.get("signals") is not None
    assert "combined" in state.get("signals", {})


def test_process_candle_buy_signal(state, warm_buffer, sample_candle):
    """Test that a strong buy signal results in a position."""
    raw, featured = warm_buffer
    engine = TradingEngine(mode="paper", state=state)
    engine.raw_buffer = raw.copy()
    engine.candle_buffer = featured

    # Model predicts big rise -> buy signal
    model = MagicMock()
    model.predict.return_value = np.array([55000.0])  # +10% above current
    engine.model = model

    engine._setup_strategies()
    engine._setup_executor()

    # Override combined strategy to always buy
    from bot.strategy import CombinedStrategy, Signal
    mock_combined = MagicMock(spec=CombinedStrategy)
    mock_combined.evaluate.return_value = (
        Signal(1, 0.9, "combined"),
        {"xgboost": Signal(1, 0.9, "xgboost"), "pattern": Signal(1, 0.6, "pattern"), "cluster": Signal(1, 0.7, "cluster")},
    )
    engine.combined_strategy = mock_combined

    engine._process_candle(sample_candle)

    # Should have opened a position
    pos = state.get_position()
    assert pos is not None
    assert pos["side"] == "long"


def test_engine_state_file_persists(tmp_path):
    state_file = str(tmp_path / "persist_test.json")
    state = BotState(initial_capital=10000, state_file=state_file)
    state.set("status", "running")

    # Reload from file
    state2 = BotState(state_file=state_file)
    assert state2.get("status") == "running"
