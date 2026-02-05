"""Data collection module — fetches BTC historical and real-time data from Binance."""

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

from config import load_api_keys, SYMBOL, DEFAULT_INTERVAL, HISTORICAL_DAYS


def create_client(keys: dict[str, str] | None = None) -> Client:
    """Create an authenticated Binance client."""
    if keys is None:
        keys = load_api_keys()
    return Client(keys["public_key"], keys["private_key"])


def get_historical_klines(
    symbol: str = SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    days: int = HISTORICAL_DAYS,
    client: Client | None = None,
) -> pd.DataFrame:
    """Fetch historical kline/candlestick data from Binance.

    Returns a DataFrame with columns: date, open, high, low, close, volume.
    Handles API throttling with exponential backoff.
    """
    if client is None:
        client = create_client()

    start_str = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%d %b %Y")
    klines = []
    retries = 0
    max_retries = 5

    while retries < max_retries:
        try:
            raw = client.get_historical_klines(symbol, interval, start_str)
            klines = raw
            break
        except BinanceAPIException as e:
            if e.code == -1003:  # rate limit
                wait = 2 ** retries
                print(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                retries += 1
            else:
                raise
        except Exception:
            wait = 2 ** retries
            time.sleep(wait)
            retries += 1

    if not klines:
        raise RuntimeError(f"Failed to fetch klines after {max_retries} retries")

    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])

    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    return df[["date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def get_realtime_price(symbol: str = SYMBOL, client: Client | None = None) -> dict:
    """Get the current price and 24h ticker for a symbol.

    Returns dict with keys: symbol, price, volume_24h, high_24h, low_24h.
    """
    if client is None:
        client = create_client()

    ticker = client.get_ticker(symbol=symbol)
    return {
        "symbol": symbol,
        "price": float(ticker["lastPrice"]),
        "volume_24h": float(ticker["volume"]),
        "high_24h": float(ticker["highPrice"]),
        "low_24h": float(ticker["lowPrice"]),
    }
