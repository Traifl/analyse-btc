"""Configuration module — loads API keys from token.txt and defines project constants."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(BASE_DIR, "token.txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")


def load_api_keys(path: str = TOKEN_FILE) -> dict[str, str]:
    """Parse token.txt and return {'public_key': ..., 'private_key': ...}."""
    keys = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, value = line.split("=", 1)
                key = key.strip()
                if key in ("public_key", "private_key"):
                    keys[key] = value.strip()
    if "public_key" not in keys or "private_key" not in keys:
        raise ValueError(f"Missing API keys in {path}")
    return keys


# --- Project constants ---
SYMBOL = "BTCUSDT"
INTERVALS = {"1h": "1h", "4h": "4h"}
DEFAULT_INTERVAL = "1h"
HISTORICAL_DAYS = 365  # 1 year of data by default

# Clustering
N_CLUSTERS = 4

# Regression
FORECAST_HORIZON = 24  # predict 24 periods ahead
TEST_SIZE = 0.2

# Backtesting
INITIAL_CAPITAL = 10_000.0
TRADE_FEE_PCT = 0.001  # 0.1% Binance fee
