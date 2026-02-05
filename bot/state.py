"""Bot state module — thread-safe persistent state for positions, trades, equity."""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from config import INITIAL_CAPITAL, OUTPUT_DIR


class BotState:
    """Thread-safe state container for the trading bot.

    Stores positions, trade history, equity curve, candle buffer,
    and current signals. Persists to JSON on every mutation.
    """

    def __init__(self, initial_capital: float = INITIAL_CAPITAL, state_file: str | None = None):
        self._lock = threading.Lock()
        self._state_file = state_file or os.path.join(OUTPUT_DIR, "bot_state.json")
        self._state: dict[str, Any] = {
            "status": "stopped",
            "mode": "paper",
            "initial_capital": initial_capital,
            "capital": initial_capital,
            "position": None,  # {"side": "long", "entry_price": ..., "quantity": ..., "entry_time": ...}
            "trades": [],      # [{"time": ..., "type": "BUY"/"SELL", "price": ..., "quantity": ..., "pnl": ...}]
            "equity_history": [],  # [{"time": ..., "equity": ...}]
            "current_price": 0.0,
            "signals": {},     # {"xgboost": ..., "pattern": ..., "cluster": ..., "combined": ...}
            "current_cluster": None,
            "current_patterns": {},
            "last_update": None,
            "candle_count": 0,
            "error": None,
        }
        self._load()

    def _load(self) -> None:
        """Load state from disk if it exists."""
        if os.path.exists(self._state_file):
            try:
                with open(self._state_file, "r") as f:
                    saved = json.load(f)
                self._state.update(saved)
            except (json.JSONDecodeError, IOError):
                pass  # Start fresh if file is corrupted

    def _save(self) -> None:
        """Persist state to disk."""
        os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
        with open(self._state_file, "w") as f:
            json.dump(self._state, f, indent=2, default=str)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._state[key] = value
            self._state["last_update"] = datetime.now(timezone.utc).isoformat()
            self._save()

    def to_dict(self) -> dict:
        """Return a snapshot of the full state."""
        with self._lock:
            return dict(self._state)

    def get_position(self) -> dict | None:
        with self._lock:
            return self._state["position"]

    def open_position(self, side: str, price: float, quantity: float) -> None:
        """Open a new position."""
        with self._lock:
            self._state["position"] = {
                "side": side,
                "entry_price": price,
                "quantity": quantity,
                "entry_time": datetime.now(timezone.utc).isoformat(),
            }
            self._state["last_update"] = datetime.now(timezone.utc).isoformat()
            self._save()

    def close_position(self, price: float) -> float:
        """Close current position and return realized PnL."""
        with self._lock:
            pos = self._state["position"]
            if pos is None:
                return 0.0

            if pos["side"] == "long":
                pnl = (price - pos["entry_price"]) * pos["quantity"]
            else:
                pnl = (pos["entry_price"] - price) * pos["quantity"]

            self._state["capital"] += pnl + pos["entry_price"] * pos["quantity"]
            self._state["position"] = None
            self._state["last_update"] = datetime.now(timezone.utc).isoformat()
            self._save()
            return pnl

    def add_trade(self, trade_type: str, price: float, quantity: float, pnl: float = 0.0) -> None:
        """Record a trade in history."""
        with self._lock:
            self._state["trades"].append({
                "time": datetime.now(timezone.utc).isoformat(),
                "type": trade_type,
                "price": price,
                "quantity": quantity,
                "pnl": pnl,
            })
            self._state["last_update"] = datetime.now(timezone.utc).isoformat()
            self._save()

    def update_equity(self, price: float) -> None:
        """Record current portfolio equity."""
        with self._lock:
            pos = self._state["position"]
            if pos is not None:
                equity = self._state["capital"] + pos["quantity"] * price
            else:
                equity = self._state["capital"]

            self._state["equity_history"].append({
                "time": datetime.now(timezone.utc).isoformat(),
                "equity": equity,
            })
            self._state["current_price"] = price
            self._state["last_update"] = datetime.now(timezone.utc).isoformat()
            self._save()

    def update_signals(self, signals: dict) -> None:
        """Update current signal values from all strategies."""
        with self._lock:
            self._state["signals"] = signals
            self._state["last_update"] = datetime.now(timezone.utc).isoformat()
            self._save()

    def reset(self, initial_capital: float | None = None) -> None:
        """Reset state for a fresh run."""
        with self._lock:
            cap = initial_capital or self._state["initial_capital"]
            self._state.update({
                "status": "stopped",
                "capital": cap,
                "initial_capital": cap,
                "position": None,
                "trades": [],
                "equity_history": [],
                "current_price": 0.0,
                "signals": {},
                "current_cluster": None,
                "current_patterns": {},
                "last_update": datetime.now(timezone.utc).isoformat(),
                "candle_count": 0,
                "error": None,
            })
            self._save()
