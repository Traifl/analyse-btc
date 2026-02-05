"""Regression module — builds predictive models for BTC price forecasting."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from config import FORECAST_HORIZON, TEST_SIZE, OUTPUT_DIR


FEATURE_COLS = [
    "ma_7", "ma_25", "ma_99", "rsi",
    "bb_upper", "bb_middle", "bb_lower", "bb_width",
    "macd", "macd_signal", "macd_hist",
    "atr", "returns", "volatility", "volume_norm",
]


def prepare_features(
    df: pd.DataFrame, horizon: int = FORECAST_HORIZON
) -> tuple[pd.DataFrame, pd.Series]:
    """Create feature matrix X and target y for regression.

    Target y is the close price `horizon` periods ahead.

    Args:
        df: DataFrame with technical features.
        horizon: Number of periods to forecast ahead.

    Returns:
        Tuple (X, y) with aligned indices, NaN rows dropped.
    """
    df = df.copy()
    df["target"] = df["close"].shift(-horizon)

    available = [c for c in FEATURE_COLS if c in df.columns]
    data = df[available + ["target"]].dropna()

    X = data[available]
    y = data["target"]
    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgboost",
    test_size: float = TEST_SIZE,
) -> tuple:
    """Train a regression model and return (model, X_test, y_test, X_train, y_train).

    Args:
        X: Feature matrix.
        y: Target series.
        model_type: 'xgboost' or 'random_forest'.
        test_size: Fraction for test split.

    Returns:
        Tuple (model, X_test, y_test, X_train, y_train).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False,  # time series: no shuffle
    )

    if model_type == "xgboost":
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            verbosity=0,
        )
    elif model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    return model, X_test, y_test, X_train, y_train


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Evaluate the model with RMSE, MAE, R².

    Returns dict with metrics.
    """
    y_pred = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }


def cross_validate_model(
    model, X: pd.DataFrame, y: pd.Series, cv: int = 5
) -> dict[str, float]:
    """Run cross-validation and return average scores.

    Note: For time series, this uses standard CV (acceptable for comparison).
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    rmse_scores = np.sqrt(-scores)
    return {
        "cv_rmse_mean": float(rmse_scores.mean()),
        "cv_rmse_std": float(rmse_scores.std()),
    }


def plot_predictions(
    y_test: pd.Series,
    y_pred: np.ndarray,
    save_path: str | None = None,
) -> None:
    """Plot actual vs predicted prices."""
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "predictions.png")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Time series comparison
    axes[0].plot(range(len(y_test)), y_test.values, label="Actual", alpha=0.8)
    axes[0].plot(range(len(y_pred)), y_pred, label="Predicted", alpha=0.8)
    axes[0].set_title("BTC Price — Actual vs Predicted")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend()

    # Scatter plot
    axes[1].scatter(y_test.values, y_pred, s=10, alpha=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("Prediction Scatter Plot")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Prediction plot saved to {save_path}")
