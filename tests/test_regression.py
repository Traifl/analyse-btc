"""Tests for regression module."""

import pytest
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.regression import prepare_features, train_model, evaluate_model


@pytest.fixture
def featured_df():
    """DataFrame with enough data for regression training."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 40000 + np.cumsum(np.random.randn(n) * 100)

    df = pd.DataFrame({
        "close": close,
        "ma_7": pd.Series(close).rolling(7).mean().values,
        "ma_25": pd.Series(close).rolling(25).mean().values,
        "ma_99": pd.Series(close).rolling(99).mean().values,
        "rsi": np.random.rand(n) * 100,
        "bb_upper": close + 500,
        "bb_middle": close,
        "bb_lower": close - 500,
        "bb_width": np.full(n, 1000.0),
        "macd": np.random.randn(n) * 100,
        "macd_signal": np.random.randn(n) * 80,
        "macd_hist": np.random.randn(n) * 20,
        "atr": np.abs(np.random.randn(n) * 200) + 50,
        "returns": np.random.randn(n) * 0.01,
        "volatility": np.abs(np.random.randn(n) * 0.005) + 0.001,
        "volume_norm": np.random.rand(n) * 2 + 0.5,
    }, index=dates)
    return df


def test_prepare_features_shape(featured_df):
    X, y = prepare_features(featured_df, horizon=24)
    assert len(X) == len(y)
    assert len(X) > 0
    assert "target" not in X.columns


def test_prepare_features_target_shift(featured_df):
    X, y = prepare_features(featured_df, horizon=1)
    # target should be close shifted by -1, so last row of close is NaN target
    assert len(y) < len(featured_df)


def test_train_model_xgboost(featured_df):
    X, y = prepare_features(featured_df, horizon=24)
    model, X_test, y_test, X_train, y_train = train_model(X, y, model_type="xgboost")
    assert len(X_test) > 0
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_train_model_random_forest(featured_df):
    X, y = prepare_features(featured_df, horizon=24)
    model, X_test, y_test, _, _ = train_model(X, y, model_type="random_forest")
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_train_model_invalid_type(featured_df):
    X, y = prepare_features(featured_df, horizon=24)
    with pytest.raises(ValueError, match="Unknown model_type"):
        train_model(X, y, model_type="invalid")


def test_evaluate_model_metrics(featured_df):
    X, y = prepare_features(featured_df, horizon=24)
    model, X_test, y_test, _, _ = train_model(X, y)
    metrics = evaluate_model(model, X_test, y_test)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
