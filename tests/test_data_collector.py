"""Tests for data_collector module — uses mocked Binance client."""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_collector import get_historical_klines, get_realtime_price


# Sample raw kline data as Binance returns it
SAMPLE_KLINE = [
    1609459200000, "29000.0", "29500.0", "28800.0", "29300.0", "1234.5",
    1609462800000, "36000000.0", 5000, "600.0", "17400000.0", "0",
]


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_historical_klines.return_value = [SAMPLE_KLINE, SAMPLE_KLINE]
    client.get_ticker.return_value = {
        "lastPrice": "29300.0",
        "volume": "1234.5",
        "highPrice": "29500.0",
        "lowPrice": "28800.0",
    }
    return client


def test_get_historical_klines_returns_dataframe(mock_client):
    df = get_historical_klines(client=mock_client, days=7)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(df) == 2


def test_get_historical_klines_types(mock_client):
    df = get_historical_klines(client=mock_client, days=7)
    assert df["open"].dtype == float
    assert df["close"].dtype == float
    assert df["volume"].dtype == float
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_get_historical_klines_values(mock_client):
    df = get_historical_klines(client=mock_client, days=7)
    row = df.iloc[0]
    assert row["open"] == 29000.0
    assert row["high"] == 29500.0
    assert row["close"] == 29300.0


def test_get_realtime_price(mock_client):
    result = get_realtime_price(client=mock_client)
    assert result["price"] == 29300.0
    assert result["volume_24h"] == 1234.5
    assert result["symbol"] == "BTCUSDT"


def test_get_historical_klines_retry_on_empty():
    """If the client always returns empty, it should raise RuntimeError."""
    client = MagicMock()
    client.get_historical_klines.return_value = []
    with pytest.raises(RuntimeError, match="Failed to fetch"):
        get_historical_klines(client=client, days=1)
