"""Tests for clustering module."""

import pytest
import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.clustering import prepare_cluster_features, perform_clustering, interpret_clusters


@pytest.fixture
def featured_df():
    """Create a DataFrame with technical features for clustering tests."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 40000 + np.cumsum(np.random.randn(n) * 100)
    df = pd.DataFrame({
        "open": close - 50,
        "high": close + 100,
        "low": close - 100,
        "close": close,
        "volume": np.random.rand(n) * 1000 + 100,
        "returns": np.random.randn(n) * 0.01,
        "volatility": np.abs(np.random.randn(n) * 0.005) + 0.001,
        "rsi": np.random.rand(n) * 100,
        "volume_norm": np.random.rand(n) * 2 + 0.5,
        "atr": np.abs(np.random.randn(n) * 200) + 50,
    }, index=dates)
    return df


def test_prepare_cluster_features_shape(featured_df):
    scaled, scaler = prepare_cluster_features(featured_df)
    assert scaled.shape[1] == 5
    assert len(scaled) == len(featured_df)


def test_prepare_cluster_features_standardized(featured_df):
    scaled, _ = prepare_cluster_features(featured_df)
    # Mean should be ~0, std ~1
    assert abs(scaled["returns"].mean()) < 0.1
    assert abs(scaled["returns"].std() - 1.0) < 0.1


def test_perform_clustering_kmeans(featured_df):
    result = perform_clustering(featured_df, method="kmeans", n_clusters=3)
    assert "cluster" in result.columns
    unique_clusters = result["cluster"].dropna().unique()
    assert len(unique_clusters) == 3


def test_perform_clustering_dbscan(featured_df):
    result = perform_clustering(featured_df, method="dbscan")
    assert "cluster" in result.columns


def test_perform_clustering_invalid_method(featured_df):
    with pytest.raises(ValueError, match="Unknown method"):
        perform_clustering(featured_df, method="invalid")


def test_interpret_clusters(featured_df):
    result = perform_clustering(featured_df, method="kmeans", n_clusters=3)
    interp = interpret_clusters(result)
    assert isinstance(interp, dict)
    assert len(interp) == 3
    for v in interp.values():
        assert isinstance(v, str)
