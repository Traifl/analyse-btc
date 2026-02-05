"""Trading engine — main loop connecting WebSocket, strategies, risk manager, executor."""

import logging
import threading
import time
import os
import sys

import pandas as pd
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from binance.client import Client

from config import load_api_keys, SYMBOL, OUTPUT_DIR
from src.data_collector import create_client, get_historical_klines
from src.preprocessing import clean_data, add_technical_features

from bot.state import BotState
from bot.strategy import XGBoostStrategy, PatternStrategy, ClusterStrategy, CombinedStrategy, Signal
from bot.risk_manager import RiskManager
from bot.executor import PaperExecutor, LiveExecutor

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
WARMUP_CANDLES = 150  # enough for MA99 + some buffer


class TradingEngine:
    """Main trading engine that connects all components.

    Flow per candle:
    1. Receive new kline from WebSocket (or polling fallback)
    2. Update candle buffer + recalculate indicators
    3. Run all strategies -> combined signal
    4. Risk manager approves/rejects
    5. Executor places trade if approved
    6. Update state for dashboard
    """

    def __init__(
        self,
        mode: str = "paper",
        symbol: str = SYMBOL,
        interval: str = "1h",
        state: BotState | None = None,
        weights: dict[str, float] | None = None,
        threshold: float | None = None,
    ):
        self.mode = mode
        self.symbol = symbol
        self.interval = interval
        self.state = state or BotState()
        self.weights = weights
        self.threshold = threshold

        self._running = False
        self._thread: threading.Thread | None = None

        # Loaded during start()
        self.client: Client | None = None
        self.model = None
        self.combined_strategy: CombinedStrategy | None = None
        self.risk_manager = RiskManager()
        self.executor = None
        self.candle_buffer: pd.DataFrame | None = None

    def _load_model(self):
        """Load pre-trained XGBoost model from models/ directory."""
        model_path = os.path.join(MODELS_DIR, "xgboost_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run run_bot.py first to train the model."
            )
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

    def _warmup(self):
        """Fetch historical candles to warm up indicators."""
        logger.info(f"Warming up with {WARMUP_CANDLES} historical candles...")
        df = get_historical_klines(
            symbol=self.symbol, interval=self.interval,
            days=30, client=self.client,
        )
        df = clean_data(df)
        df = add_technical_features(df)
        df = df.dropna()

        # Keep last WARMUP_CANDLES
        if len(df) > WARMUP_CANDLES:
            df = df.iloc[-WARMUP_CANDLES:]

        self.candle_buffer = df
        self.state.set("candle_count", len(df))
        logger.info(f"Warm-up done: {len(df)} candles in buffer")

    def _setup_strategies(self):
        """Initialize all strategy objects."""
        strategies = [
            XGBoostStrategy(self.model),
            PatternStrategy(),
            ClusterStrategy(),
        ]
        self.combined_strategy = CombinedStrategy(
            strategies,
            weights=self.weights,
            threshold=self.threshold,
        )

    def _setup_executor(self):
        """Create the appropriate executor based on mode."""
        if self.mode == "live":
            self.executor = LiveExecutor(self.client, self.symbol)
        else:
            self.executor = PaperExecutor()

    def _process_candle(self, candle: dict):
        """Process a single new candle through the full pipeline."""
        try:
            # Build new row
            new_row = pd.DataFrame([{
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
                "volume": float(candle["volume"]),
            }], index=[pd.Timestamp(candle["close_time"], unit="ms")])

            # Append to buffer and recalculate features
            raw = pd.concat([self.candle_buffer[["open", "high", "low", "close", "volume"]], new_row])
            featured = add_technical_features(raw)
            featured = featured.dropna()

            # Trim buffer
            if len(featured) > WARMUP_CANDLES:
                featured = featured.iloc[-WARMUP_CANDLES:]
            self.candle_buffer = featured

            current_price = float(candle["close"])
            self.state.set("current_price", current_price)
            self.state.set("candle_count", len(featured))

            # Run strategies
            combined_signal, individual_signals = self.combined_strategy.evaluate(featured)

            # Update state with signal info
            sig_dict = {}
            for name, sig in individual_signals.items():
                sig_dict[name] = {"direction": sig.direction, "confidence": sig.confidence}
            sig_dict["combined"] = {"direction": combined_signal.direction, "confidence": combined_signal.confidence}
            self.state.update_signals(sig_dict)

            # Risk check
            approved, qty = self.risk_manager.check_risk(combined_signal, current_price, self.state)

            if approved is not None and qty > 0:
                if approved.direction == 1:
                    self.executor.execute_buy(current_price, qty, self.state)
                elif approved.direction == -1:
                    self.executor.execute_sell(current_price, qty, self.state)

            # Always check SL/TP even if no new signal
            if approved is None:
                sl_tp = self.risk_manager.check_stop_loss_take_profit(self.state, current_price)
                if sl_tp is not None:
                    pos = self.state.get_position()
                    if pos:
                        self.executor.execute_sell(current_price, pos["quantity"], self.state)

            # Update equity
            self.state.update_equity(current_price)
            self.state.set("error", None)

        except Exception as e:
            logger.error(f"Error processing candle: {e}", exc_info=True)
            self.state.set("error", str(e))

    def _polling_loop(self):
        """Polling-based main loop (fallback if WebSocket not available).

        Fetches latest candles periodically and processes new ones.
        """
        logger.info("Starting polling loop...")
        last_candle_time = self.candle_buffer.index[-1] if len(self.candle_buffer) > 0 else None

        while self._running:
            try:
                # Fetch recent klines
                df = get_historical_klines(
                    symbol=self.symbol, interval=self.interval,
                    days=2, client=self.client,
                )

                for _, row in df.iterrows():
                    candle_time = row["date"]
                    if last_candle_time is not None and candle_time <= last_candle_time:
                        continue

                    candle = {
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"],
                        "close_time": int(candle_time.timestamp() * 1000),
                    }
                    self._process_candle(candle)
                    last_candle_time = candle_time
                    logger.info(f"Processed candle at {candle_time}: close=${row['close']:,.2f}")

            except Exception as e:
                logger.error(f"Polling error: {e}", exc_info=True)
                self.state.set("error", str(e))

            # Sleep interval: check every 60s for 1h candles
            for _ in range(60):
                if not self._running:
                    break
                time.sleep(1)

    def start(self):
        """Start the trading engine."""
        logger.info(f"Starting trading engine (mode={self.mode}, symbol={self.symbol})")

        # Initialize
        keys = load_api_keys()
        self.client = create_client(keys)
        self._load_model()
        self._setup_strategies()
        self._setup_executor()
        self._warmup()

        self.state.set("status", "running")
        self.state.set("mode", self.mode)
        self._running = True

        # Start polling in background thread
        self._thread = threading.Thread(target=self._polling_loop, daemon=True)
        self._thread.start()
        logger.info("Trading engine started")

    def stop(self):
        """Stop the trading engine."""
        logger.info("Stopping trading engine...")
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        self.state.set("status", "stopped")
        logger.info("Trading engine stopped")

    @property
    def is_running(self) -> bool:
        return self._running
