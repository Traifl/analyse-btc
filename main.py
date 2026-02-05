"""Main orchestrator — runs all analysis steps and produces output report."""

import os
import sys
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_api_keys, SYMBOL, OUTPUT_DIR, FORECAST_HORIZON
from src.data_collector import create_client, get_historical_klines, get_realtime_price
from src.preprocessing import clean_data, add_technical_features
from src.clustering import perform_clustering, interpret_clusters, plot_clusters
from src.patterns import detect_patterns, generate_pattern_report
from src.regression import prepare_features, train_model, evaluate_model, cross_validate_model, plot_predictions
from src.backtesting import generate_signals, backtest_strategy, plot_backtest, generate_backtest_report


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Step 1: Data Collection ----
    print("=" * 60)
    print("STEP 1: Data Collection")
    print("=" * 60)

    keys = load_api_keys()
    client = create_client(keys)

    print(f"Fetching historical 1H data for {SYMBOL}...")
    df_1h = get_historical_klines(symbol=SYMBOL, interval="1h", days=365, client=client)
    print(f"  -> {len(df_1h)} candles retrieved")

    print(f"Fetching historical 4H data for {SYMBOL}...")
    df_4h = get_historical_klines(symbol=SYMBOL, interval="4h", days=365, client=client)
    print(f"  -> {len(df_4h)} candles retrieved")

    realtime = get_realtime_price(symbol=SYMBOL, client=client)
    print(f"  Current price: ${realtime['price']:,.2f}")

    # Use 1H data for the main analysis pipeline
    df = df_1h

    # ---- Step 2: Preprocessing ----
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing")
    print("=" * 60)

    df = clean_data(df)
    print(f"  Cleaned: {len(df)} rows, no NaN")

    df = add_technical_features(df)
    print(f"  Added technical features: {df.shape[1]} columns total")

    # Drop warm-up NaN rows from indicators
    df = df.dropna()
    print(f"  After dropping indicator warm-up: {len(df)} rows")

    # ---- Step 3: Clustering ----
    print("\n" + "=" * 60)
    print("STEP 3: Clustering")
    print("=" * 60)

    df = perform_clustering(df, method="kmeans", n_clusters=4)
    interpretations = interpret_clusters(df)
    print("  Cluster interpretations:")
    for cid, desc in interpretations.items():
        count = (df["cluster"] == cid).sum()
        print(f"    Cluster {cid}: {desc} ({count} periods)")

    plot_clusters(df, save_path=os.path.join(OUTPUT_DIR, "clusters.png"))

    # ---- Step 4: Pattern Analysis ----
    print("\n" + "=" * 60)
    print("STEP 4: Pattern Analysis")
    print("=" * 60)

    df = detect_patterns(df)
    pattern_report = generate_pattern_report(df)
    print(pattern_report)

    # ---- Step 5: Regression ----
    print("\n" + "=" * 60)
    print("STEP 5: Regression")
    print("=" * 60)

    X, y = prepare_features(df, horizon=FORECAST_HORIZON)
    print(f"  Features: {X.shape[1]}, Samples: {len(X)}")

    # XGBoost
    print("\n  Training XGBoost...")
    xgb_model, X_test, y_test, X_train, y_train = train_model(X, y, model_type="xgboost")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
    print(f"  XGBoost — RMSE: ${xgb_metrics['rmse']:,.2f}  MAE: ${xgb_metrics['mae']:,.2f}  R²: {xgb_metrics['r2']:.4f}")

    xgb_cv = cross_validate_model(xgb_model, X, y, cv=5)
    print(f"  XGBoost CV — RMSE: ${xgb_cv['cv_rmse_mean']:,.2f} ± ${xgb_cv['cv_rmse_std']:,.2f}")

    # RandomForest for comparison
    print("\n  Training RandomForest...")
    rf_model, _, _, _, _ = train_model(X, y, model_type="random_forest")
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print(f"  RandomForest — RMSE: ${rf_metrics['rmse']:,.2f}  MAE: ${rf_metrics['mae']:,.2f}  R²: {rf_metrics['r2']:.4f}")

    # Use the best model
    best_name = "XGBoost" if xgb_metrics["rmse"] < rf_metrics["rmse"] else "RandomForest"
    best_model = xgb_model if best_name == "XGBoost" else rf_model
    best_metrics = xgb_metrics if best_name == "XGBoost" else rf_metrics
    print(f"\n  Best model: {best_name}")

    y_pred = best_model.predict(X_test)
    plot_predictions(y_test, y_pred, save_path=os.path.join(OUTPUT_DIR, "predictions.png"))

    # ---- Step 6: Backtesting ----
    print("\n" + "=" * 60)
    print("STEP 6: Backtesting")
    print("=" * 60)

    signals = generate_signals(df, best_model.predict(X), X.index)
    bt_results = backtest_strategy(signals)
    bt_report = generate_backtest_report(bt_results)
    print(bt_report)

    plot_backtest(bt_results, save_path=os.path.join(OUTPUT_DIR, "backtest.png"))

    # ---- Final Report ----
    print("\n" + "=" * 60)
    print("Generating final report...")
    print("=" * 60)

    report = f"""# BTC Analysis Report — {SYMBOL}

## 1. Data Collection
- **1H candles**: {len(df_1h)}
- **4H candles**: {len(df_4h)}
- **Current price**: ${realtime['price']:,.2f}
- **24H volume**: {realtime['volume_24h']:,.2f} BTC

## 2. Preprocessing
- {df.shape[1]} features computed (MAs, RSI, Bollinger, MACD, ATR, etc.)
- {len(df)} clean data points after indicator warm-up

## 3. Clustering (K-Means, k=4)
"""
    for cid, desc in interpretations.items():
        count = (df["cluster"] == cid).sum()
        report += f"- **Cluster {cid}**: {desc} ({count} periods)\n"

    report += f"""
![Clusters](clusters.png)

{pattern_report}

## 5. Regression — {best_name}

| Metric | Value |
|--------|-------|
| RMSE | ${best_metrics['rmse']:,.2f} |
| MAE | ${best_metrics['mae']:,.2f} |
| R² | {best_metrics['r2']:.4f} |
| CV RMSE | ${xgb_cv['cv_rmse_mean']:,.2f} ± ${xgb_cv['cv_rmse_std']:,.2f} |

### Model Comparison

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| XGBoost | ${xgb_metrics['rmse']:,.2f} | ${xgb_metrics['mae']:,.2f} | {xgb_metrics['r2']:.4f} |
| RandomForest | ${rf_metrics['rmse']:,.2f} | ${rf_metrics['mae']:,.2f} | {rf_metrics['r2']:.4f} |

![Predictions](predictions.png)

{bt_report}

![Backtest](backtest.png)
"""

    report_path = os.path.join(OUTPUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nFinal report saved to {report_path}")
    print("All outputs saved to output/")
    print("Done!")


if __name__ == "__main__":
    main()
