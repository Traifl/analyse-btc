"""Preprocessing module — cleans data and adds technical indicators."""

import pandas as pd
import numpy as np
import ta


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw OHLCV data: handle missing values, set datetime index, sort.

    Args:
        df: DataFrame with columns date, open, high, low, close, volume.

    Returns:
        Cleaned DataFrame with DatetimeIndex.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Forward-fill then back-fill small gaps
    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # Drop any remaining rows with NaN
    df = df.dropna(subset=numeric_cols).reset_index(drop=True)

    df = df.set_index("date")
    return df


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to a cleaned OHLCV DataFrame.

    Adds: MA(7, 25, 99), RSI(14), Bollinger Bands, MACD, ATR(14),
    returns, log_returns, volatility(20).

    Args:
        df: DataFrame with DatetimeIndex and columns open, high, low, close, volume.

    Returns:
        DataFrame with additional feature columns.
    """
    df = df.copy()

    # Moving averages
    df["ma_7"] = df["close"].rolling(window=7).mean()
    df["ma_25"] = df["close"].rolling(window=25).mean()
    df["ma_99"] = df["close"].rolling(window=99).mean()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()

    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # ATR
    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()

    # Returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Rolling volatility (20 periods)
    df["volatility"] = df["returns"].rolling(window=20).std()

    # Volume normalized
    df["volume_norm"] = df["volume"] / df["volume"].rolling(window=20).mean()

    return df
