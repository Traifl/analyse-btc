"""Pattern analysis module — detects recurring price patterns (range, breakout, retracement)."""

import pandas as pd
import numpy as np
import os

from config import OUTPUT_DIR


# Default thresholds — can be overridden
DEFAULT_THRESHOLDS = {
    "range_atr_percentile": 25,       # ATR below this percentile = range
    "breakout_bb_margin": 0.0,        # close above bb_upper = breakout up
    "retracement_fib_levels": [0.236, 0.382, 0.5, 0.618, 0.786],
    "retracement_tolerance": 0.01,    # 1% tolerance around fib levels
    "volume_spike_factor": 2.0,       # volume_norm > this = volume spike
}


def detect_range(df: pd.DataFrame, thresholds: dict | None = None) -> pd.Series:
    """Detect range/consolidation periods (low ATR).

    Returns boolean Series: True where market is in a range.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    atr_threshold = df["atr"].quantile(t["range_atr_percentile"] / 100)
    return df["atr"] < atr_threshold


def detect_breakout(df: pd.DataFrame, thresholds: dict | None = None) -> pd.Series:
    """Detect breakout patterns (price crossing Bollinger Bands).

    Returns Series with values: 'breakout_up', 'breakout_down', or None.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    margin = t["breakout_bb_margin"]

    conditions = [
        df["close"] > df["bb_upper"] + margin,
        df["close"] < df["bb_lower"] - margin,
    ]
    choices = ["breakout_up", "breakout_down"]
    return pd.Series(
        np.select(conditions, choices, default=None),
        index=df.index,
        dtype="object",
    )


def detect_retracement(df: pd.DataFrame, window: int = 50, thresholds: dict | None = None) -> pd.Series:
    """Detect Fibonacci retracement levels.

    Compares current price to recent high/low and checks if it sits near a Fibonacci level.
    Returns Series with the closest fib level or NaN.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    fib_levels = t["retracement_fib_levels"]
    tolerance = t["retracement_tolerance"]

    rolling_high = df["high"].rolling(window=window).max()
    rolling_low = df["low"].rolling(window=window).min()
    price_range = rolling_high - rolling_low

    retracement = pd.Series(np.nan, index=df.index)

    for fib in fib_levels:
        fib_price = rolling_high - fib * price_range
        near_fib = (abs(df["close"] - fib_price) / df["close"]) < tolerance
        retracement = retracement.where(~near_fib, fib)

    return retracement


def detect_volume_spike(df: pd.DataFrame, thresholds: dict | None = None) -> pd.Series:
    """Detect periods of abnormally high volume.

    Returns boolean Series.
    """
    t = thresholds or DEFAULT_THRESHOLDS
    return df["volume_norm"] > t["volume_spike_factor"]


def detect_patterns(df: pd.DataFrame, thresholds: dict | None = None) -> pd.DataFrame:
    """Run all pattern detectors and add columns to the DataFrame.

    Added columns: is_range, breakout, fib_retracement, volume_spike.
    """
    df = df.copy()
    required = ["atr", "bb_upper", "bb_lower", "close", "high", "low", "volume_norm"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["is_range"] = detect_range(df, thresholds)
    df["breakout"] = detect_breakout(df, thresholds)
    df["fib_retracement"] = detect_retracement(df, thresholds=thresholds)
    df["volume_spike"] = detect_volume_spike(df, thresholds)
    return df


def generate_pattern_report(df: pd.DataFrame) -> str:
    """Generate a markdown summary of detected patterns.

    Returns markdown string with pattern statistics.
    """
    total = len(df)
    range_count = df["is_range"].sum()
    breakout_up = (df["breakout"] == "breakout_up").sum()
    breakout_down = (df["breakout"] == "breakout_down").sum()
    fib_count = df["fib_retracement"].notna().sum()
    vol_spike = df["volume_spike"].sum()

    report = f"""# Pattern Analysis Report

## Summary (total periods: {total})

| Pattern | Count | Percentage |
|---------|-------|------------|
| Range / Consolidation | {range_count} | {range_count/total*100:.1f}% |
| Breakout Up | {breakout_up} | {breakout_up/total*100:.1f}% |
| Breakout Down | {breakout_down} | {breakout_down/total*100:.1f}% |
| Fibonacci Retracement | {fib_count} | {fib_count/total*100:.1f}% |
| Volume Spike | {vol_spike} | {vol_spike/total*100:.1f}% |

## Observations

- **Range periods** represent low-volatility consolidation (ATR below 25th percentile).
- **Breakouts** occur when price crosses Bollinger Bands boundaries.
- **Fibonacci retracements** detected near standard levels (23.6%, 38.2%, 50%, 61.8%, 78.6%).
- **Volume spikes** indicate periods where volume exceeds 2x the 20-period average.
"""
    return report
