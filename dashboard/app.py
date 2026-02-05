"""Streamlit dashboard — real-time monitoring of the BTC trading bot."""

import json
import os
import sys
import time

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import OUTPUT_DIR, SYMBOL

STATE_FILE = os.path.join(OUTPUT_DIR, "bot_state.json")

# Page config
st.set_page_config(
    page_title=f"BTC Trading Bot — {SYMBOL}",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_state() -> dict:
    """Load bot state from JSON file."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def signal_emoji(direction: int) -> str:
    if direction == 1:
        return "🟢 BUY"
    elif direction == -1:
        return "🔴 SELL"
    return "⚪ HOLD"


def main():
    state = load_state()

    if not state:
        st.title("BTC Trading Bot")
        st.warning("No bot state found. Start the bot with: `python run_bot.py`")
        st.stop()

    # ── Header ──────────────────────────────────────────────────
    status = state.get("status", "unknown")
    mode = state.get("mode", "paper").upper()
    current_price = state.get("current_price", 0)
    capital = state.get("capital", 0)
    initial_capital = state.get("initial_capital", 0)

    # Calculate total equity
    pos = state.get("position")
    if pos:
        equity = capital + pos["quantity"] * current_price
    else:
        equity = capital

    pnl = equity - initial_capital
    pnl_pct = (pnl / initial_capital * 100) if initial_capital > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        status_color = "🟢" if status == "running" else "🔴"
        st.metric("Status", f"{status_color} {status.upper()}")
    with col2:
        st.metric("Mode", f"{'📋' if mode == 'PAPER' else '💰'} {mode}")
    with col3:
        st.metric("BTC Price", f"${current_price:,.2f}")
    with col4:
        st.metric("Portfolio", f"${equity:,.2f}", delta=f"{pnl_pct:+.2f}%")
    with col5:
        st.metric("P&L", f"${pnl:,.2f}", delta=f"{pnl:+,.2f}")

    st.divider()

    # ── Signals Panel ───────────────────────────────────────────
    st.subheader("Trading Signals")
    signals = state.get("signals", {})

    if signals:
        sig_cols = st.columns(len(signals))
        for i, (name, sig) in enumerate(signals.items()):
            with sig_cols[i]:
                direction = sig.get("direction", 0)
                confidence = sig.get("confidence", 0)
                emoji = signal_emoji(direction)
                st.metric(
                    label=name.upper(),
                    value=emoji,
                    delta=f"Confidence: {confidence:.0%}",
                )
    else:
        st.info("No signals yet. Waiting for first candle...")

    st.divider()

    # ── Two-column layout ──────────────────────────────────────
    left_col, right_col = st.columns([2, 1])

    with left_col:
        # ── Equity Curve ────────────────────────────────────────
        st.subheader("Equity Curve")
        equity_history = state.get("equity_history", [])
        if equity_history:
            eq_df = pd.DataFrame(equity_history)
            eq_df["time"] = pd.to_datetime(eq_df["time"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq_df["time"], y=eq_df["equity"],
                mode="lines", name="Portfolio Value",
                line=dict(color="#2196F3", width=2),
                fill="tozeroy", fillcolor="rgba(33, 150, 243, 0.1)",
            ))
            fig.add_hline(
                y=initial_capital,
                line_dash="dash", line_color="gray",
                annotation_text="Initial Capital",
            )
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis_title="Value (USD)",
                xaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No equity history yet.")

    with right_col:
        # ── Position Info ───────────────────────────────────────
        st.subheader("Current Position")
        if pos:
            entry_price = pos["entry_price"]
            quantity = pos["quantity"]
            unrealized_pnl = (current_price - entry_price) * quantity
            unrealized_pct = (current_price - entry_price) / entry_price * 100

            st.metric("Side", f"📈 {pos['side'].upper()}")
            st.metric("Entry Price", f"${entry_price:,.2f}")
            st.metric("Quantity", f"{quantity:.6f} BTC")
            st.metric(
                "Unrealized P&L",
                f"${unrealized_pnl:,.2f}",
                delta=f"{unrealized_pct:+.2f}%",
            )
            st.caption(f"Opened: {pos.get('entry_time', 'N/A')}")
        else:
            st.info("No open position")

        # ── Stats ───────────────────────────────────────────────
        st.subheader("Stats")
        trades = state.get("trades", [])
        n_trades = len(trades)
        winning = [t for t in trades if t.get("pnl", 0) > 0]
        win_rate = len(winning) / n_trades * 100 if n_trades > 0 else 0

        st.metric("Total Trades", n_trades)
        st.metric("Win Rate", f"{win_rate:.1f}%")
        st.metric("Candles Processed", state.get("candle_count", 0))

    st.divider()

    # ── Trade History ───────────────────────────────────────────
    st.subheader("Trade History")
    trades = state.get("trades", [])
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_df["time"] = pd.to_datetime(trades_df["time"])
        trades_df = trades_df.sort_values("time", ascending=False)

        # Color PnL
        st.dataframe(
            trades_df[["time", "type", "price", "quantity", "pnl"]].head(50),
            use_container_width=True,
            column_config={
                "time": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm"),
                "type": st.column_config.TextColumn("Type"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "quantity": st.column_config.NumberColumn("Qty", format="%.6f"),
                "pnl": st.column_config.NumberColumn("P&L", format="$%.2f"),
            },
        )
    else:
        st.info("No trades yet.")

    # ── Error Display ───────────────────────────────────────────
    error = state.get("error")
    if error:
        st.error(f"Bot Error: {error}")

    # ── Footer ──────────────────────────────────────────────────
    last_update = state.get("last_update", "N/A")
    st.caption(f"Last update: {last_update} | Auto-refresh every 5s")

    # Auto-refresh
    time.sleep(5)
    st.rerun()


if __name__ == "__main__":
    main()
