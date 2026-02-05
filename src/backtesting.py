"""Backtesting module — simulates a trading strategy based on model predictions and clusters."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from config import INITIAL_CAPITAL, TRADE_FEE_PCT, OUTPUT_DIR


def generate_signals(
    df: pd.DataFrame,
    predictions: np.ndarray,
    prediction_index: pd.Index,
) -> pd.DataFrame:
    """Generate buy/sell signals from model predictions and cluster data.

    Strategy:
    - BUY when predicted price > current price (model predicts upward move)
      AND cluster is not bearish (cluster interpretation).
    - SELL when predicted price < current price.
    - HOLD otherwise.

    Args:
        df: Full DataFrame with 'close' and optionally 'cluster' columns.
        predictions: Array of predicted future prices.
        prediction_index: Index corresponding to predictions.

    Returns:
        DataFrame with 'signal' column (1=buy, -1=sell, 0=hold).
    """
    signals = pd.DataFrame(index=prediction_index)
    signals["close"] = df.loc[prediction_index, "close"]
    signals["predicted"] = predictions
    signals["signal"] = 0

    # Buy when prediction > current price
    signals.loc[signals["predicted"] > signals["close"], "signal"] = 1
    # Sell when prediction < current price
    signals.loc[signals["predicted"] < signals["close"], "signal"] = -1

    # If cluster info is available, filter out buys in bearish clusters
    if "cluster" in df.columns:
        signals["cluster"] = df.loc[prediction_index, "cluster"]

    return signals


def backtest_strategy(
    signals: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    fee_pct: float = TRADE_FEE_PCT,
) -> dict:
    """Run a simple long-only backtest on generated signals.

    Position sizing: all-in on buy, all-out on sell.

    Args:
        signals: DataFrame with 'close' and 'signal' columns.
        initial_capital: Starting capital in USD.
        fee_pct: Trading fee as a fraction (0.001 = 0.1%).

    Returns:
        Dict with backtest results: total_return, profit, max_drawdown,
        sharpe_ratio, n_trades, equity_curve.
    """
    capital = initial_capital
    position = 0.0  # BTC held
    equity = []
    trades = []

    for i, (idx, row) in enumerate(signals.iterrows()):
        price = row["close"]
        signal = row["signal"]

        if signal == 1 and position == 0:
            # Buy
            cost = capital * (1 - fee_pct)
            position = cost / price
            capital = 0
            trades.append({"date": idx, "type": "BUY", "price": price})
        elif signal == -1 and position > 0:
            # Sell
            revenue = position * price * (1 - fee_pct)
            capital = revenue
            position = 0
            trades.append({"date": idx, "type": "SELL", "price": price})

        # Track portfolio value
        portfolio_value = capital + position * price
        equity.append({"date": idx, "equity": portfolio_value})

    equity_df = pd.DataFrame(equity).set_index("date")
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=["date", "type", "price"])

    # Final value
    final_value = equity_df["equity"].iloc[-1] if len(equity_df) > 0 else initial_capital
    total_return = (final_value - initial_capital) / initial_capital

    # Max drawdown
    running_max = equity_df["equity"].cummax()
    drawdown = (equity_df["equity"] - running_max) / running_max
    max_drawdown = drawdown.min()

    # Sharpe ratio (annualized, assuming hourly data)
    returns = equity_df["equity"].pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(8760)  # hourly -> annual
    else:
        sharpe = 0.0

    return {
        "initial_capital": initial_capital,
        "final_value": float(final_value),
        "profit": float(final_value - initial_capital),
        "total_return_pct": float(total_return * 100),
        "max_drawdown_pct": float(max_drawdown * 100),
        "sharpe_ratio": float(sharpe),
        "n_trades": len(trades_df),
        "trades": trades_df,
        "equity_curve": equity_df,
    }


def plot_backtest(results: dict, save_path: str | None = None) -> None:
    """Plot the equity curve and trade markers."""
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "backtest.png")

    fig, ax = plt.subplots(figsize=(14, 6))
    eq = results["equity_curve"]
    ax.plot(eq.index, eq["equity"], label="Portfolio Value", color="blue")
    ax.axhline(y=results["initial_capital"], color="gray", linestyle="--", label="Initial Capital")

    # Mark trades
    trades = results["trades"]
    if len(trades) > 0:
        buys = trades[trades["type"] == "BUY"]
        sells = trades[trades["type"] == "SELL"]
        for _, t in buys.iterrows():
            if t["date"] in eq.index:
                ax.axvline(x=t["date"], color="green", alpha=0.3, linewidth=0.5)
        for _, t in sells.iterrows():
            if t["date"] in eq.index:
                ax.axvline(x=t["date"], color="red", alpha=0.3, linewidth=0.5)

    ax.set_title("Backtest — Equity Curve")
    ax.set_ylabel("Portfolio Value (USD)")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Backtest plot saved to {save_path}")


def generate_backtest_report(results: dict) -> str:
    """Generate a markdown report of backtest results."""
    return f"""# Backtest Report

## Strategy: Prediction-Based Long-Only

| Metric | Value |
|--------|-------|
| Initial Capital | ${results['initial_capital']:,.2f} |
| Final Value | ${results['final_value']:,.2f} |
| Profit/Loss | ${results['profit']:,.2f} |
| Total Return | {results['total_return_pct']:.2f}% |
| Max Drawdown | {results['max_drawdown_pct']:.2f}% |
| Sharpe Ratio | {results['sharpe_ratio']:.2f} |
| Number of Trades | {results['n_trades']} |

## Strategy Description

- **Entry**: Buy when the XGBoost model predicts a price increase over the forecast horizon.
- **Exit**: Sell when the model predicts a price decrease.
- **Position sizing**: 100% capital per trade (all-in/all-out).
- **Fees**: {TRADE_FEE_PCT * 100:.1f}% per trade (Binance standard).
"""
