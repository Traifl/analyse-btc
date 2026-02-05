"""Strategy module — individual strategies and weighted ensemble voting."""

import numpy as np
import pandas as pd
from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import add_technical_features
from src.clustering import perform_clustering, interpret_clusters
from src.patterns import detect_patterns
from src.regression import FEATURE_COLS


@dataclass
class Signal:
    """A trading signal from a strategy."""
    direction: int    # +1 buy, -1 sell, 0 hold
    confidence: float # 0.0 to 1.0
    name: str         # strategy name


class XGBoostStrategy:
    """Predict future price with XGBoost and compare to current price."""

    def __init__(self, model, feature_cols: list[str] | None = None):
        self.model = model
        self.feature_cols = feature_cols or FEATURE_COLS

    def evaluate(self, df: pd.DataFrame) -> Signal:
        """Generate signal from the latest row of featured data."""
        available = [c for c in self.feature_cols if c in df.columns]
        row = df[available].iloc[[-1]].dropna(axis=1)
        if row.empty or len(row.columns) < len(available) * 0.5:
            return Signal(0, 0.0, "xgboost")

        prediction = self.model.predict(row)[0]
        current_price = df["close"].iloc[-1]
        diff_pct = (prediction - current_price) / current_price

        if diff_pct > 0.005:  # >0.5% predicted rise
            confidence = min(abs(diff_pct) * 20, 1.0)  # scale to 0-1
            return Signal(1, confidence, "xgboost")
        elif diff_pct < -0.005:  # >0.5% predicted fall
            confidence = min(abs(diff_pct) * 20, 1.0)
            return Signal(-1, confidence, "xgboost")
        return Signal(0, 0.3, "xgboost")


class PatternStrategy:
    """Generate signals based on detected chart patterns."""

    def evaluate(self, df: pd.DataFrame) -> Signal:
        """Signal from pattern detection on latest row."""
        required = ["atr", "bb_upper", "bb_lower", "close", "high", "low", "volume_norm"]
        if not all(c in df.columns for c in required):
            return Signal(0, 0.0, "pattern")

        patterned = detect_patterns(df)
        last = patterned.iloc[-1]

        if last.get("breakout") == "breakout_up" and last.get("volume_spike"):
            return Signal(1, 0.9, "pattern")  # strong breakout with volume
        elif last.get("breakout") == "breakout_up":
            return Signal(1, 0.6, "pattern")
        elif last.get("breakout") == "breakout_down" and last.get("volume_spike"):
            return Signal(-1, 0.9, "pattern")
        elif last.get("breakout") == "breakout_down":
            return Signal(-1, 0.6, "pattern")
        elif last.get("is_range"):
            return Signal(0, 0.5, "pattern")  # consolidation = hold
        return Signal(0, 0.3, "pattern")


class ClusterStrategy:
    """Generate signals based on current market cluster/regime."""

    def evaluate(self, df: pd.DataFrame) -> Signal:
        """Signal from clustering the latest data point."""
        required = ["returns", "volatility", "rsi", "volume_norm", "atr"]
        if not all(c in df.columns for c in required):
            return Signal(0, 0.0, "cluster")

        clustered = perform_clustering(df, method="kmeans", n_clusters=4)
        interpretations = interpret_clusters(clustered)
        current_cluster = clustered["cluster"].iloc[-1]

        if pd.isna(current_cluster):
            return Signal(0, 0.0, "cluster")

        label = interpretations.get(int(current_cluster), "")

        if "Bullish" in label:
            return Signal(1, 0.7, "cluster")
        elif "Bearish" in label:
            return Signal(-1, 0.7, "cluster")
        elif "volatility" in label.lower():
            return Signal(0, 0.4, "cluster")  # high vol = caution
        return Signal(0, 0.3, "cluster")


class CombinedStrategy:
    """Weighted ensemble of multiple strategies."""

    DEFAULT_WEIGHTS = {"xgboost": 0.5, "pattern": 0.25, "cluster": 0.25}
    DEFAULT_THRESHOLD = 0.3  # minimum weighted score to trigger trade

    def __init__(
        self,
        strategies: list,
        weights: dict[str, float] | None = None,
        threshold: float | None = None,
    ):
        self.strategies = strategies
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.threshold = threshold if threshold is not None else self.DEFAULT_THRESHOLD

    def evaluate(self, df: pd.DataFrame) -> tuple[Signal, dict[str, Signal]]:
        """Evaluate all strategies and return combined signal + individual signals.

        Returns:
            Tuple of (combined_signal, {strategy_name: individual_signal}).
        """
        individual: dict[str, Signal] = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for strategy in self.strategies:
            sig = strategy.evaluate(df)
            individual[sig.name] = sig
            w = self.weights.get(sig.name, 0.0)
            weighted_sum += sig.direction * sig.confidence * w
            total_weight += w

        if total_weight > 0:
            score = weighted_sum / total_weight
        else:
            score = 0.0

        if score > self.threshold:
            direction = 1
        elif score < -self.threshold:
            direction = -1
        else:
            direction = 0

        combined = Signal(direction, abs(score), "combined")
        return combined, individual
