"""Clustering module — segments BTC price behaviors using K-Means or DBSCAN."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import N_CLUSTERS, OUTPUT_DIR
import os


CLUSTER_FEATURES = ["returns", "volatility", "rsi", "volume_norm", "atr"]


def prepare_cluster_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Extract and scale features for clustering.

    Args:
        df: DataFrame with technical features already added.

    Returns:
        Tuple of (scaled features DataFrame, fitted scaler).
    """
    features = df[CLUSTER_FEATURES].dropna()
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(features),
        index=features.index,
        columns=CLUSTER_FEATURES,
    )
    return scaled, scaler


def perform_clustering(
    df: pd.DataFrame,
    method: str = "kmeans",
    n_clusters: int = N_CLUSTERS,
) -> pd.DataFrame:
    """Cluster the data and add a 'cluster' column.

    Args:
        df: DataFrame with technical features.
        method: 'kmeans' or 'dbscan'.
        n_clusters: Number of clusters (for kmeans).

    Returns:
        DataFrame with an additional 'cluster' column.
    """
    df = df.copy()
    scaled, _ = prepare_cluster_features(df)

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
    elif method == "dbscan":
        model = DBSCAN(eps=1.0, min_samples=10)
        labels = model.fit_predict(scaled)
    else:
        raise ValueError(f"Unknown method: {method}")

    df.loc[scaled.index, "cluster"] = labels
    df["cluster"] = df["cluster"].astype("Int64")
    return df


def interpret_clusters(df: pd.DataFrame) -> dict[int, str]:
    """Produce a human-readable interpretation for each cluster.

    Returns dict mapping cluster_id -> description string.
    """
    interpretations = {}
    grouped = df.dropna(subset=["cluster"]).groupby("cluster")

    for cluster_id, group in grouped:
        avg_ret = group["returns"].mean()
        avg_vol = group["volatility"].mean()
        avg_rsi = group["rsi"].mean()

        if avg_ret > 0.001 and avg_rsi > 55:
            label = "Bullish (positive returns, high RSI)"
        elif avg_ret < -0.001 and avg_rsi < 45:
            label = "Bearish (negative returns, low RSI)"
        elif avg_vol > df["volatility"].dropna().quantile(0.75):
            label = "High volatility regime"
        else:
            label = "Range / consolidation"

        interpretations[int(cluster_id)] = label
    return interpretations


def plot_clusters(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Generate cluster visualizations and save to disk.

    Creates two plots: price colored by cluster, and a 2D scatter of returns vs volatility.
    """
    data = df.dropna(subset=["cluster"])
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "clusters.png")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Price timeline colored by cluster
    for cid in sorted(data["cluster"].unique()):
        mask = data["cluster"] == cid
        axes[0].scatter(
            data.index[mask], data["close"][mask],
            label=f"Cluster {cid}", s=4, alpha=0.7,
        )
    axes[0].set_title("BTC Price — Clusters over Time")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend()

    # Plot 2: Returns vs Volatility scatter
    scatter = axes[1].scatter(
        data["returns"], data["volatility"],
        c=data["cluster"], cmap="viridis", s=8, alpha=0.6,
    )
    axes[1].set_xlabel("Returns")
    axes[1].set_ylabel("Volatility")
    axes[1].set_title("Returns vs Volatility by Cluster")
    plt.colorbar(scatter, ax=axes[1], label="Cluster")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Cluster plot saved to {save_path}")
