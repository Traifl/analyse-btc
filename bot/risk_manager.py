"""Risk manager — position sizing, stop-loss, take-profit, trade cooldown."""

from datetime import datetime, timedelta, timezone

from bot.state import BotState
from bot.strategy import Signal
from config import TRADE_FEE_PCT


class RiskManager:
    """Controls risk exposure for the trading bot.

    Enforces: max risk per trade, stop-loss, take-profit,
    max concurrent positions, cooldown between trades.
    """

    def __init__(
        self,
        max_risk_pct: float = 0.02,     # risk 2% of capital per trade
        stop_loss_pct: float = 0.03,     # -3% stop loss
        take_profit_pct: float = 0.05,   # +5% take profit
        max_positions: int = 1,
        cooldown_hours: int = 4,
        fee_pct: float = TRADE_FEE_PCT,
    ):
        self.max_risk_pct = max_risk_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_positions = max_positions
        self.cooldown_hours = cooldown_hours
        self.fee_pct = fee_pct

    def check_stop_loss_take_profit(self, state: BotState, current_price: float) -> Signal | None:
        """Check if current position should be closed due to SL/TP.

        Returns a SELL signal if triggered, None otherwise.
        """
        pos = state.get_position()
        if pos is None:
            return None

        entry = pos["entry_price"]
        if pos["side"] == "long":
            pnl_pct = (current_price - entry) / entry
            if pnl_pct <= -self.stop_loss_pct:
                return Signal(-1, 1.0, "stop_loss")
            if pnl_pct >= self.take_profit_pct:
                return Signal(-1, 1.0, "take_profit")
        return None

    def check_cooldown(self, state: BotState) -> bool:
        """Return True if enough time has passed since last trade."""
        trades = state.get("trades", [])
        if not trades:
            return True
        last_trade_time = datetime.fromisoformat(trades[-1]["time"])
        elapsed = datetime.now(timezone.utc) - last_trade_time
        return elapsed >= timedelta(hours=self.cooldown_hours)

    def calculate_position_size(self, capital: float, price: float) -> float:
        """Calculate BTC quantity to buy based on max risk percentage.

        Returns quantity in BTC.
        """
        risk_amount = capital * self.max_risk_pct
        # Position size = risk_amount / (stop_loss_pct * price)
        position_value = risk_amount / self.stop_loss_pct
        # Cap at available capital minus fees
        max_value = capital * (1 - self.fee_pct)
        position_value = min(position_value, max_value)
        return position_value / price

    def check_risk(self, signal: Signal, current_price: float, state: BotState) -> tuple[Signal | None, float]:
        """Evaluate whether a signal should be executed given risk constraints.

        Returns:
            Tuple of (approved_signal_or_None, quantity).
            None means the trade is rejected.
        """
        # First check SL/TP — these override everything
        sl_tp = self.check_stop_loss_take_profit(state, current_price)
        if sl_tp is not None:
            pos = state.get_position()
            return sl_tp, pos["quantity"] if pos else 0.0

        # If signal is hold, nothing to do
        if signal.direction == 0:
            return None, 0.0

        # BUY checks
        if signal.direction == 1:
            # Already have a position?
            if state.get_position() is not None:
                return None, 0.0
            # Cooldown check
            if not self.check_cooldown(state):
                return None, 0.0
            # Calculate size
            capital = state.get("capital", 0)
            if capital <= 0:
                return None, 0.0
            qty = self.calculate_position_size(capital, current_price)
            return signal, qty

        # SELL checks
        if signal.direction == -1:
            pos = state.get_position()
            if pos is None:
                return None, 0.0  # nothing to sell
            return signal, pos["quantity"]

        return None, 0.0
