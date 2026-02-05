"""Trade executor — paper and live execution modes."""

import logging
from abc import ABC, abstractmethod

from binance.client import Client
from binance.exceptions import BinanceAPIException

from bot.state import BotState
from config import TRADE_FEE_PCT, SYMBOL

logger = logging.getLogger(__name__)


class Executor(ABC):
    """Base class for trade executors."""

    @abstractmethod
    def execute_buy(self, price: float, quantity: float, state: BotState) -> bool:
        """Execute a buy order. Returns True on success."""
        ...

    @abstractmethod
    def execute_sell(self, price: float, quantity: float, state: BotState) -> bool:
        """Execute a sell order. Returns True on success."""
        ...


class PaperExecutor(Executor):
    """Simulated execution for paper trading."""

    def __init__(self, fee_pct: float = TRADE_FEE_PCT):
        self.fee_pct = fee_pct

    def execute_buy(self, price: float, quantity: float, state: BotState) -> bool:
        cost = quantity * price * (1 + self.fee_pct)
        capital = state.get("capital", 0)
        if cost > capital:
            quantity = capital * (1 - self.fee_pct) / price
            cost = quantity * price * (1 + self.fee_pct)

        state.set("capital", capital - cost)
        state.open_position("long", price, quantity)
        state.add_trade("BUY", price, quantity)
        logger.info(f"[PAPER] BUY {quantity:.6f} BTC @ ${price:,.2f} (cost: ${cost:,.2f})")
        return True

    def execute_sell(self, price: float, quantity: float, state: BotState) -> bool:
        pnl = state.close_position(price)
        revenue = quantity * price * (1 - self.fee_pct)
        state.add_trade("SELL", price, quantity, pnl=pnl)
        logger.info(f"[PAPER] SELL {quantity:.6f} BTC @ ${price:,.2f} (PnL: ${pnl:,.2f})")
        return True


class LiveExecutor(Executor):
    """Real execution via Binance API."""

    def __init__(self, client: Client, symbol: str = SYMBOL, fee_pct: float = TRADE_FEE_PCT):
        self.client = client
        self.symbol = symbol
        self.fee_pct = fee_pct

    def execute_buy(self, price: float, quantity: float, state: BotState) -> bool:
        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side="BUY",
                type="MARKET",
                quantity=f"{quantity:.6f}",
            )
            fill_price = float(order["fills"][0]["price"]) if order.get("fills") else price
            fill_qty = float(order["executedQty"])

            cost = fill_qty * fill_price
            capital = state.get("capital", 0)
            state.set("capital", capital - cost)
            state.open_position("long", fill_price, fill_qty)
            state.add_trade("BUY", fill_price, fill_qty)
            logger.info(f"[LIVE] BUY {fill_qty:.6f} BTC @ ${fill_price:,.2f}")
            return True
        except BinanceAPIException as e:
            logger.error(f"[LIVE] BUY failed: {e}")
            state.set("error", f"BUY failed: {e}")
            return False

    def execute_sell(self, price: float, quantity: float, state: BotState) -> bool:
        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side="SELL",
                type="MARKET",
                quantity=f"{quantity:.6f}",
            )
            fill_price = float(order["fills"][0]["price"]) if order.get("fills") else price
            fill_qty = float(order["executedQty"])

            pnl = state.close_position(fill_price)
            state.add_trade("SELL", fill_price, fill_qty, pnl=pnl)
            logger.info(f"[LIVE] SELL {fill_qty:.6f} BTC @ ${fill_price:,.2f} (PnL: ${pnl:,.2f})")
            return True
        except BinanceAPIException as e:
            logger.error(f"[LIVE] SELL failed: {e}")
            state.set("error", f"SELL failed: {e}")
            return False
